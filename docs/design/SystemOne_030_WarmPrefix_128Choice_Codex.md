# SystemOne 共享前缀预热迁移与 128 Choice 扩展任务书

日期：2026-10-29
用途：交给本地 Codex，在现有 vLLM 0.30 Qwen Decisions 回移分支上实施、验证和交付。
本文件提供源码分析和实施设计；已下载官方 Qwen3.6-27B 固定版本的 tokenizer 并完成 128 标签的真实分词及回复边界验证。未修改目标仓库、未运行模型推理或 NPU 性能实验。

## 1. 执行目标与固定基线

完成两个独立、可分别回归的改动：

1. 将官方 PR #61192 的公共前缀预热机制迁入现有 0.30 分支，减少混合 Attention/GDN 模型在多题冷请求中的重复 Prefill。
2. 将单道 choice 的上限扩为 128，每题仍只有一次首 token 概率读出；保留已有 1–26 项请求的标签、prompt 和概率契约。

**版本关系需要准确表述：当前已把 0.32 的 Decisions API 回移到 0.30；预热优化来自 0.32 tag 之后的 main，属于本次补迁。原始 v0.32.0 tag 没有 `_warm_shared_prefix()`。**

| 用途 | 仓库、分支或固定 SHA |
| --- | --- |
| 目标仓库 | `F00L42/vllm`，`codex/v0.30.0-qwen-decisions` |
| 本次审计目标 HEAD | `ea2e3575d07bf49fdefa1e74ceb349896ff74f8d` |
| vLLM 0.30 底座 | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| 原 Decisions API 来源 v0.32.0 | `5df63a7203cdb9328d824a27b682e77c7b9928be` |
| 预热优化来源 | PR #61192，`ee22f682ab0d2139ae69e3cdbb762a565a166813` |
| 本次查看的官方 main | `fa3b9d1377219247060d3c8b49eae199c9e7e154` |
| 目标部署文档中的 Ascend 配套 | v0.30.0rc2，`8fde6d4e335fe300e2b949aab0e31659cce0811d` |
| 旧自研服务参考 | `F00L42/vllm-ascend@4d7786dbb2e15be660b73a4f63a2331bf9815d91`，仅用于对照 |

本地工作区可能已有后续修改。执行前读 AGENTS.md、git status、HEAD 和现有差异，以真实本地状态为准；保留用户已有工作，不 reset 到上述 SHA。目标主要路径是：

```text
vllm/entrypoints/generate/structured_decisions/
```

最新 main 的 `vllm/entrypoints/openai/decisions/` 路径不能直接覆盖到目标。此次优先修改 vLLM Python 层，保留现有 Ascend、torch、torch-npu、CANN 依赖；仅在确认具体后端能力缺口后增加最小 Ascend 补丁。[S1][S2][S3]

## 2. 上游预热机制的源码阅读顺序

| 阅读位置 | 要理解的事情 |
| --- | --- |
| main `strategies.py` 的 `common_prefix_len`、`MIN_WARM_PREFIX_TOKENS` | 对完整渲染后的 token 序列求 LCP；阈值为 128 tokens |
| `NextTokenStrategy._warm_shared_prefix` | 判断是否值得预热，提交内部 producer 并等待结束 |
| `NextTokenStrategy.read` | 先完成各题渲染、标签校验，再 await warm，最后并发各题 |
| 目标 `MambaManager.get_num_blocks_to_allocate` | 为什么同一步产生的 recurrent state 不能马上给另一个请求复用 |
| 目标 `get_mamba_prefill_checkpoint_position` 与 `is_mamba_prefill_checkpoint_valid` | 预热结束后究竟有哪些位置能留下可恢复的状态 |
| 目标 `label_reads.py` | 分支仍是普通 generate；按指定 token ID 顺序收集概率 |

当前 main 的函数约在 `strategies.py:371`，调用处约在 `:489`；目标策略的 `read` 在 `:278`，并发读出在 `:316`。使用固定 SHA 链接阅读，行号只作定位。[S2][S4][S5]

### 2.1 精确执行过程

以下为机制伪代码，不是可直接复制的补丁：

```python
rendered = render_and_validate_all_questions()
prompts = [item.full_prompt_token_ids for item in rendered]

if (
    len(prompts) >= 2
    and model_is_hybrid
    and prefix_caching_enabled
):
    shared_length = longest_common_token_prefix_length(prompts)
    if shared_length >= 128:
        await run_to_completion(
            prompt_ids=prompts[0][:shared_length],
            max_output_tokens=1,
            temperature=0,
            same_cache_salt=True,
            same_lora=True,
            same_priority=True,
            same_trace_context=True,
        )

answers = await submit_original_full_prompts_concurrently(rendered)
```

关键含义：

- LCP 来自**完整的 chat-template 渲染结果**。不能用 state 字符串长度，也不能把 state 单独 encode 后假定它就是完整输入的前缀。
- 公共部分可能包含 system instruction、state 以及各题相同的起始文本。
- producer 使用普通 `engine_client.generate`；内部 request ID 后缀是 `-prefix`。
- producer 的采样参数不请求标签 logprobs。生成的那个 token 仅用于正常走完一次请求，结果丢弃，不拼入任何分支。
- 各题仍提交原始完整输入，实际少算多少由引擎 APC 决定。
- `max_tokens=1` 的普通自回归路径从 Prefill 末位置产生首 token，不要求额外的下一轮 Decode；长 Prefill 仍可拆成多个 forward。投机配置的实际执行步骤另行观测。
- 当前函数没有显式取整到物理 block，没有额外添加一个输入 token，没有要求 eager 或关闭 async。
- 128 是“额外预热请求是否值得”的经验阈值，与 128 个 choice 候选的容量无关。[S2][S3]

**Qwen3.6-27B 的纯文本请求满足 hybrid 条件。** 这里 hybrid 指文本主干混合使用 GDN 与 Full Attention，与请求是否带图片独立。官方配置是 64 层，每组 3 层 linear_attention 加 1 层 full_attention，共 48 层 GDN、16 层 Full Attention；vLLM 的 Qwen3_5ForConditionalGeneration 类继承 IsHybrid。纯文本仍经过这个混合文本主干。是否实际 warm，还取决于多题、APC 和 LCP 阈值。[S14][S17]

### 2.2 为什么对 GDN 混合模型有帮助

Attention 的 KV 和 GDN 的 recurrent state 是不同类型的缓存。GDN 处理完前缀后，会得到代表该前缀的状态；后续分支需要从这个状态恢复，并拥有可独立更新的工作状态。

目标 0.30 的 `MambaManager.get_num_blocks_to_allocate` 检查 `cached_blocks_this_step`：若消费者要复用的 recurrent block 是当前调度步才产生的，就返回超过可用块数的分配需求，使这个消费者暂不进入该步。这样可以避免复制尚未计算完成的状态。

**预热改变了依赖顺序：先让一个 producer 计算完成，再让各分支查找已生成的状态。** 同时，producer 的终点位于分叉附近，为该位置留下可复用 checkpoint 创造条件。它不是新增一套 KV 张量管理器，也不把 N 道题融合成同一个分类头。[S5]

不要删除上述 same-step gate 来“提高并发”。该约束保护 recurrent state 的正确性。

### 2.3 预热长度与实际命中长度必须分开

设完整公共前缀长度为 P，分支 i 的问题后缀长度为 S_i，实际命中长度为 C_i。

官方 producer 输入长度是 P；它并不承诺 C_i=P。目标 0.30 已有内部 checkpoint、partial-tail cache 等机制，但可导出的状态受以下条件影响：

- 实际 Attention/Mamba cache spec、hash/prefix-match 粒度；
- backend 的 `prefill_checkpoint_alignment`；
- 分块边界、checkpoint 导出和恢复支持；
- state copy / copy-on-write、保留和淘汰策略；
- 投机配置及跨 DP、外部缓存路由。

例如目标的 checkpoint 位置 helper 在非 Eagle 路径计算：

```text
floor((num_tokens - 1) / hash_block_size) * hash_block_size
```

Eagle 路径还会再减一个 hash unit；另一个 helper 继续检查 backend alignment 等条件。这个公式只描述该内部 checkpoint 路径，**不能直接当作所有分支最终 cache hit 的公式**。完整物理块等其他路径也可能参与命中。[S6]

因此，第一阶段应保留上游“完整 LCP 预热”的机制，再通过 Ascend 实测验证。若目标后端没有所需的内部 checkpoint 导出能力，要定位具体缺口；只有经过证据验证后，才考虑旧方案那种“producer 截断到已知有效边界”的独立后备实现。

### 2.4 生命周期、用量和局限

上游 warm 循环只消费结果，不执行异常回退；generate 抛错时不会继续提交分支。取消和 abort 主要由底层 AsyncLLM 生命周期处理。它没有额外的跨 HTTP producer 合并、缓存固定引用或“热缓存则跳过”判断。[S2]

迁移时建议补足以下**本地健壮性增强**，并在来源合同中明确标注：

- 正常消费完且确认成功，才启动分支；空流、error/abort 终态不能被当作预热成功。
- 取消遵循目标 AsyncLLM 的清理契约，必要时关闭生成器或 abort；避免双重 abort，保证清理幂等。
- 不用宽泛 except 把真正的模型错误吞掉后重试。
- 能力检查不满足时，在 producer 提交前沿用原分支路径，并记录跳过原因。
- 不在进程内用“这个 state 曾预热过”的集合代表缓存仍存在；引擎可能已经淘汰它。

producer 的内部工作量不进入现有每题 `QuestionRead` 的 API usage；引擎指标会看到额外请求。应保留现有外部 usage 语义，单独记录 warm 指标，性能统一看完整 HTTP 时延。

热缓存下 producer 也不一定只花一次调度往返：generate 仍需计算末位置取得 logits，结合 checkpoint 和对齐可能重算末尾一段。应同时记录 producer 自身的实际命中 H；忽略抢占重算时，一般工作量表达为 `(P-H)+sum(P-C_i+S_i)`。不能因为 state 曾经热过，就把 producer 成本记为零。

## 3. 效果：工作量推导与公开实测

### 3.1 计算量推导

以下只估算新计算的输入 token，不等价于实际 FLOPs 或时延；Full Attention 后缀仍要读取前缀 KV，恢复状态也有开销。

冷缓存、原路径完全不复用时：

```text
W_before = N*P + sum(S_i)
W_after  = P + sum(P - C_i + S_i)
saved    = sum(C_i) - P
```

当所有 C_i=P，节省 (N-1)P；当所有 C_i=0，预热反而多计算 P。因此，“请求顺序正确”与“缓存确实复用”都要验证。

例如 P=16384、N=3、每题 S_i=256，理想全命中：

| 情况 | 新计算 token |
| --- | ---: |
| 原路径无共享 | 49920 |
| 一次预热后各题只算后缀 | 17152 |
| 减少比例 | 65.6% |

原路径本来就能命中时，应以实际调度 token 对照，不能把上述最坏基线当作现有服务的真实工作量。旧 `4d7786d` 将 producer 自身截断到 C，公式则是 C+sum(P-C+S_i)，与官方 producer=P 的情况不同。

### 3.2 PR 作者的公开测试

PR #61192 报告了以下完整 SystemOne 请求 p50，state 为 2048 tokens：

| 问题数 N | 修改前 | 修改后 | 根据数据计算的加速比 |
| --- | ---: | ---: | ---: |
| 1 | 6.4 s | 6.4 s | 1.00× |
| 4 | 25.3 s | 8.1 s | 3.12× |
| 16 | 106.5 s | 12.0 s | 8.88× |
| 64 | 419.1 s | 28.7 s | 14.60× |

环境是 Qwen3.8-27B、Qwen3_5 hybrid、4bit、MTP=3、AMD Radeon 8060S/gfx1151；使用 APC、align、prefix-match-unit=32。测试每组 3 次，属于作者给出的特定环境证据。不能把这些倍数作为 Ascend 验收承诺。

作者还报告纯 Attention 模型强行预热后略慢，因此默认只对 hybrid 生效；没有给出完整热缓存回归表。跨设备迁移必须补测热缓存和 DG 混跑。[S3]

## 4. 128 Choice：一次请求读取 128 个单 token 标签

### 4.1 最小设计

目标已有两层 26 项限制：

- `NextTokenStrategy.limits()` 的 max_options；
- `build_question()` 对 `qtype.label_set` 长度的限制。

本次只扩展 choice，保留 noul 的 yes/no 和 score 的 0–9。目标只注册三种 Qwen 策略，可先采用 choice 专属常量，不必建设通用插件或新增外部 API。

已对官方 Qwen3.6-27B 验证通过的固定码表：

```python
CHOICE_LABELS_128 = tuple(string.ascii_uppercase) + tuple("""
AA AB AC AD AE AF AG AH AI AJ AK AL AM AN AO AP AQ AR AS AT AU AV AW AX AY AZ
BA BB BC BD BE BF BG BH BI BJ BK BL BM BN BO BP BR BS BT BU BV BW BX BY
CA CB CC CD CE CF CG CH CI CK CL CM CN CO CP CR CS CT CU CV CW CX CY
DA DB DC DD DE DF DG DH DI DJ DK DL DM DN DO DP DR DS DT DU DV DW DX DY
EA EB EC ED EE
""".split())
```

即 A–Z 加 102 个确定可用的大写双字母，共 128 项。验证固定于官方 revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`、官方 chat template、`enable_thinking=False`；使用 Transformers 5.5.4 和 tokenizers 0.22.2，24 种上下文的 3072 组“完整 prompt + 标签”检查全部通过。详见第 11 节与 `qwen36_choice128_verified.json`。[S15][S16]

**此前 A–Z 加 0–101 的候选已否决。** 该 tokenizer 按单个数字预切分，`10 → [16, 15]`，`101 → [16, 15, 16]`；原候选的 92 个多位数字均不满足要求。双字母也不能按完整笛卡尔积直接分配：例如 `BQ → [33, 48]`、`BZ → [33, 57]`、`CJ → [34, 41]`，故上述固定表有意跳过它们。

约束如下：

1. 前 26 项的标签顺序不变。
2. K≤26 时保留原有 prompt 的每一个字符，尤其保留原回复指令。
3. K>26 时才改用“返回一个选项的精确标签”的回复指令，避免继续要求只能返回字母。
4. 标签只是内部编码；返回的 choice 和 probabilities 键仍是业务 option name。
5. 标签集合固定、顺序稳定；不能按请求随机挑词表 token。
6. 每项严格一个 token，不增加大小写别名，不使用多 token 标签。[S4][S7]

回复指令应通过纯函数或方法按当前 question 选择。`QUESTION_TYPES` 持有共享题型实例，禁止在请求处理过程中修改 `question.type.reply_instruction`，否则并发的大小题会互相污染 prompt。

实现时直接使用上述固定表。启动时对实际部署 tokenizer/template 复核；若文件或模板与已验证版本不同，运行附带验证脚本确认。其他模型需要独立码表验收，不能复用“Qwen”名称作保证。不要在请求时重新筛选标签，也不要退化为多次 choice。

### 4.2 必须使用真实回复边界验证

复用目标现有 `reply_label_ids()`，基于实际 chat template 和 thinking=False 的 generation prompt，要求：

- 追加某标签后，旧 prompt tail 完整保持；
- 只新增一个 token；
- 128 个标签映射到 128 个互不相同的 token ID；
- ID 不属于 special token，不越界；
- 每个真实请求的 prompt tail 与初始化时校验的 tail 一致。

现有 `reply_label_ids()` 已覆盖 tail、单 token 和集合内去重；special token 排除与 ID 范围校验需要显式补充。不能把这两项当作现有 helper 已经完成的检查。

现有 `type_label_ids()` 遍历题型标签。若使用 choice 专属常量，令 `ChoiceQuestion.label_set` 指向该常量，初始化应能验证全部 128 项；`NextTokenStrategy.limits()` 使用该集合长度。保留 `LABELS=A–Z` 作为原始基础常量，避免把 noul/score 或未来其他策略无意扩大。

采用本次全大写字母码表后，全题型的初始化映射包含 140 个不同 ID（128 个 choice，加 yes/no，再加 score 的 0–9）。128 的容量限制作用于每一次 generate 的指定 ID 列表，不作用于启动时的联合映射；每个请求只提取当前题目的标签。

验证报告至少保存：

```json
{
  "model_architecture": "...",
  "tokenizer_path_or_revision": "...",
  "tokenizer_files_sha256": "...",
  "chat_template_sha256": "...",
  "chat_template_kwargs": {"enable_thinking": false},
  "labels": ["A", "B", "..."],
  "token_ids": [0],
  "count": 128,
  "all_single_token": true,
  "all_unique": true
}
```

上面是结构示意，不能把占位内容当作验证产物。实际字段必须填写真实列表与哈希。若初始化失败，应明确报告不支持该 128 码表；对同一服务继续支持 26 项的降级如需实现，应由显式 legacy profile 控制，不能一边宣称 128 一边截断。

目标路由初始化会捕获策略初始化的 ValueError，并可能只写 debug 后禁用整个 SystemOne，随后返回通用 501。新增码表验证失败时，必须以可见日志报告失败标签、原因、模型和模板身份，并测试最终服务状态；可在策略初始化处完成，必要时最小修改路由初始化。不能让 26 项也失效却只有隐藏的调试信息。

### 4.3 目标 0.30 已有足够的读出容量

已核目标 `sampling_params.py` 中 `MAX_LOGPROB_TOKEN_IDS=128`。保留现有参数组合：

```python
SamplingParams(
    max_tokens=1,
    temperature=0.0,
    logprob_token_ids=choice_token_ids,
)
```

不要额外设置 `logprobs=128`：指定 token IDs 和普通 top-k logprobs 是不同入口，目标的默认 `max_logprobs` 不需要因此提高。已有实现会计算完整词表 logits/logprobs，再 gather 指定标签；不会为每个标签执行一次模型。[S8][S9]

注意 128 指**指定候选数**。sampled token 有独立输出位置；当 sampled token 不在候选里，底层可能返回 129 个不同 token 的条目。这不能成为把 choice 上限降到 127 的理由。上层必须按原指定 IDs 取分数，不能对返回字典的所有条目直接求和。

Ascend rc2 的当前部署路径固定 V1 runner；根据目标部署手册，V2 的指定 token logprobs 尚未支持。核查实际安装分支和 AscendSampler 继承链，不要为了 128 项把 GPU 新 runner/Triton 的缓冲实现搬到 NPU。[S1][S9]

### 4.4 概率契约和性能边界

保留目标分支已有官方语义。以 raw_logprobs 为实验基线：

```text
p_i = exp(label_logprob_i)
q_i = p_i / sum(p_j)
label_mass = sum(p_i)
confidence = max(q_i) * label_mass
```

`argmax_is_label` 仍检查真实采样 token 是否在候选内。不要增加 allowed_token_ids 或 logit_bias 来强制标签输出，否则 label_mass 和诊断含义会变化。不要把 raw_logits 当成可直接取 exp 的全词表对数概率。[S7]

保持一次 Prefill 的收益，是不增加额外分组请求与 Decode。128 项比 26 项仍有更多选项文本，需要额外 Prefill；读出 gather、响应组装也随 K 增长。不能承诺总延迟不变。

这两项改动作用于不同维度：

| 请求形态 | 分支 engine 请求数 | 满足条件时的 warm 请求数 |
| --- | ---: | ---: |
| 1 题 × 128 选项 | 1 | 0 |
| 4 题 × 26 选项 | 4 | 1 |
| 4 题 × 128 选项 | 4 | 1 |

第一阶段不引入候选专用 LM Head。它还涉及全词表归一化、label_mass、TP、量化和权重访问，是另外一个优化课题。

## 5. 本地 Codex 的实施步骤

### 步骤 A：建立可追踪基线

1. 读取本地仓库说明，记录 vLLM/Ascend 两个 HEAD、当前分支、dirty diff、实际 runner、安装版本和源码路径。
2. 对照上表确认 030 底座、032 API 来源和 warm 来源。已经存在的功能只补缺口。
3. 先运行现有离线合同与 smoke 的可运行部分，保存基线。
4. 记录实际模型、tokenizer、template、dtype、TP/DP、缓存与调度配置。没有 NPU 时继续完成实现及 CPU 检查，NPU 项标为 pending_remote。
5. 复核本地 tokenizer/template 与第 11 节固定版本。官方版本的分词验证已经完成；本地部署若无法复核，应单列“部署版本待确认”，不能将其写成已完成本地端到端验收。

### 步骤 B：回移预热编排

最小回移点是在 `read()` 中保存各题已渲染 prompt IDs，在全部校验完成后、`next_token_label_reads()` 前插入 await。

保留四个上游 gate：多题、hybrid、APC、LCP≥128。为 A/B 与回滚提供一个服务级开关；优先复用现有配置入口，若没有，新增一个经过 `vllm.envs` 注册的布尔变量即可，例如建议名 `VLLM_DECISIONS_WARM_SHARED_PREFIX`。这是拟新增配置，不是现有官方参数。首次交付可默认关闭，完成目标部署验收后再决定默认值。阈值先固定 128，避免新增多套调参入口。

透传原 salt、LoRA、priority、trace；各题完整 prompt 保持不变。验证 AsyncLLM 终态和取消清理后再实现必要增强。保持 API usage 契约，将 producer 成本放在内部日志/指标。

本轮维持目标的文本输入范围。不能把多模态输入切成 token-only producer：图像输入及其缓存身份并不由 token 切片完整表达。目标已有媒体拒绝逻辑应保留；若本地后续分支已接入媒体，本轮对该路径跳过 warm 并记录原因，另行验收。

**admission 的 n 不能机械从 N 改为 N+1。** 目标检查的是将占用的 unfinished request 槽；warm 结束后才开始 N 个分支，单父请求峰值仍为 N。累计 engine 请求数才是 N+1。检查是否需要在 warm 后重新做组级 admission，以目标引擎检查位置和并发行为为准。[S10]

先在 DP=1、无外部 KV connector 的配置验收。若部署为多 DP 或 P/D 分离，须证明 producer 与 branches 能访问同一缓存域，再扩大支持范围。不要未经验证把“不同路由仍能命中”当作事实。

### 步骤 C：验证混合缓存能力

不要默认改动 scheduler/KV manager。目标 030 已有相关机制，先检查：

- GDN backend 生成的 MambaSpec 是否声明 prefill checkpoint 数量和 alignment；
- worker 元数据是否传到算子，算子是否真实写出 checkpoint；
- Attention KV 与 recurrent state 命中边界是否一致；
- 分支是否通过独立状态或 copy-on-write 恢复，某分支推进不会污染其他分支；
- 实际 prefix-match-unit、physical block 和 retention 配置；
- warm 完成后每个分支真实 num_cached_tokens，以及调度器新增计算 token。

如果 warm 成功但 hit=0，顺着“spec声明→调度分配→runner元数据→算子导出→注册→lookup→恢复”定位，提供具体证据。不要删除同一步复用保护，也不要用提高 max_num_batched_tokens 掩盖未复用问题。

`enforce_eager` 和同步调度可作为首轮定位配置；本轮实现不得无条件要求它们。后续单独验收 FULL_DECODE_ONLY 与 async，避免为了 Decision 优化关闭 Generation 图执行。

### 步骤 D：扩展 choice

1. 加入第 4.1 节已验证的 choice 专属 128 码表，在实际部署 tokenizer 上运行复验脚本。
2. 同时修改 strategy limits 和 choice label_set。
3. 仅 K>26 改回复指令；K≤26 的 prompt IDs 必须与基线一致。
4. 保持 noul/score、概率公式、HTTP schema 和采样参数。
5. 不改 MAX_LOGPROB_TOKEN_IDS、不增加 logprobs top-k、不拆分问题。
6. 输出码表与模型/template 适用范围。

### 步骤 E：维护源码合同与文档

目标 `tools/decisions/verify_upstream.py` 当前对 `question_types.py` 整模块、`NextTokenStrategy` 整类等做 AST 相等。正常扩展会打破旧合同。[S11]

应把合同拆成：

- 未改模块/方法继续与原 032 SOURCE 严格相等。
- warm helper、调用顺序单独对应 PERF_SOURCE=ee22f682...，0.30适配和生命周期增强逐项登记。
- choice 常量、限额、K>26 prompt 属于有意扩展，用明确的符号白名单和独立合同验证。
- 保留 answer、softmax、协议、旧 prompt 等关键语义检查。
- 不要通过跳过整个策略或删除整个 question_types 检查让测试“变绿”。

部署文档还存在 `metadata.version(...).startswith("0.30.0+qwen.decisions.032")` 检查，应改为 `packaging.version.Version` 的语义比较，因为 PEP 440 会把 .032 规范化为 .32。若更新本地版本后缀，比较实际新的 expected 版本。

继续保留 `--no-deps --no-build-isolation` 的安装策略、V1 runner 约束及只启用一份 SystemOne 路由的要求；不要同时启用老 Ascend 自建 endpoint。[S1]

## 6. 文件级修改范围

| 文件或位置 | 修改目的 |
| --- | --- |
| `structured_decisions/strategies.py` | LCP、warm gate/执行、调用顺序、128 limits/标签验证 |
| `structured_decisions/question_types.py` | choice专属码表、K>26回复指令；保留原题型语义 |
| `vllm/envs.py` 或已有配置入口 | 一个预热 A/B 开关；仅在需要时增加 |
| `structured_decisions/serving.py` | 仅必要的配置/观测/admission适配；优先不改业务协议 |
| `tools/decisions/verify_upstream.py` | 原语义+性能来源+choice扩展的分层合同 |
| `test_next_token_strategy.py` | producer屏障、失败清理、元数据、128读出 |
| `test_question_types.py` | 27/127/128通过、129拒绝、旧题型兼容 |
| `test_next_token_tokenizer.py` | 实际tokenizer/template的完整码表验证 |
| `tools/decisions/run_offline_tests.py` | 必要的测试加载适配 |
| `tools/decisions/smoke_qwen.py` | 128/129边界、单题请求数、26回归 |
| 新增码表验证工具和性能脚本 | 输出可复核JSON/CSV证据 |
| `tools/decisions/README.md`、部署手册 | 新来源、能力范围、启动/回滚与验收状态 |

`protocol.py`、`api_router.py`、`label_reads.py`、sampler/KV manager 默认不改；确实发现缺口时，说明最小必要差异。

当前离线替身要有针对性调整：fake tokenizer 按字符编码，不能证明多位数字是单token；fake engine 需要 hybrid/APC 属性和无 logprob IDs 的 warm 流；原固定三项分数不能复用于128项；分支并发屏障应排除producer。不要让替身“任意字符串都算一个token”，应保留多token/重复ID拒绝测试。

## 7. 验证与性能验收

### 7.1 CPU 和协议合同

- LCP：空/单条输入按调用契约处理、相同prompt、不同首token、较短prompt、127/128阈值。
- gate：单题、非hybrid、APC关闭、短LCP均不发producer。
- producer未完成时没有分支；成功后各题可以并发。
- 渲染或标签校验失败发生于任何producer提交之前。
- warm/branches的salt、LoRA、priority、trace一致；分支token IDs与不开warm完全相同。
- producer异常、取消、空流、error/abort终态与子请求清理。
- 同时多个父HTTP请求的内部ID不冲突，不发生跨请求数据或状态混用。
- K=1、2、26、27、62、127、128通过，129为明确400；score仍2–10级，noul仍2项。
- 128候选+sampled不在集合内，全部128候选到达上层；sampled在集合内不重复累计。
- 混合K=2/26/128组批、请求移除/swap后的映射正确，无padding token污染。
- 缺失候选logprob明确失败；非法/全非有限分数不产生NaN响应。
- K≤26的labels、完整prompt IDs、response shape和usage保持原契约。

以上涉及真实采样缓冲的混合 K、swap/remove 和 padding 项，需运行可用的真实 sampler 合同测试，并在 Ascend V1 集成中复测。fake engine 的测试只能证明上层组装和映射逻辑；offline runner 通过不能把 NPU 缓冲验收标为完成。

### 7.2 NPU 数值正确性

固定同一prompt IDs、模型权重和配置，对照 warm关闭/开启：

1. 先保存重复运行的正常数值波动，预先设定比较容差。
2. 比较每个标签 logprob、候选概率、label_mass、confidence和答案。
3. 测不同后缀、相同前缀、不同salt、不同LoRA以及缓存淘汰/抢占。
4. 顺序改变分支执行次序，验证无共享GDN状态污染。
5. 单题128项与完整词表logprob reference在同一输入下逐ID对照。
6. 验证边界附近长度 U−1/U/U+1、物理块边界附近以及不同chunk预算；U来自实际配置。

### 7.3 性能实验必须拆成三组

#### 实验一：预热本身的收益

同一代码、相同prompt、相同全部启动参数，只切warm开关：

- N=1/4/16，必要时扩展64；
- P=2K/16K/32K，在模型长度和显存允许范围内；
- 逻辑冷缓存：每次使用新的cache_salt，同一父请求的producer和分支共享该salt；
- 热缓存：先完成同内容、同salt请求，再测重复请求；
- 先DP1无竞争，再加父HTTP并发及Generation混流。

盐变更保持token输入不变，适合隔离APC；还应独立预热编译/算子，避免把首次编译成本当作冷KV成本。

#### 实验二：128个数的读出增量

固定完全相同的prompt，只把指定ID数量从26变为64/128，测gather、输出处理和完整请求时间。此实验用于隔离读出成本，不用于比较不同标签集合的答案。

#### 实验三：真实128选项业务成本

真实K=26/64/128题目，记录新增prompt tokens和总延迟；做位置打乱/标签重映射的离线质量实验。增多选项使输入更长，也改变竞争分布，不宜要求不同K之间概率或confidence完全相同。

最低记录字段建议：

```text
vllm_sha, ascend_sha, model/tokenizer/template_hash
warm_enabled, N, K, lcp_tokens, warm_input_tokens, warm_cached_tokens
warm_ms, branches_ms, http_ms, per_branch_cached_tokens
per_step_scheduled_tokens, num_computed_tokens, preemptions
cache_mode, prefix_match_unit, checkpoint_alignment, retention
p50, p95, requests_per_s, questions_per_s
generation_TTFT, generation_TPOT_or_ITL
```

生产日志只保留必要聚合值；逐步调度数据使用受控debug选项，避免日志本身干扰性能。

### 7.4 完成标准

| 层面 | 完成标准 |
| --- | --- |
| 128容量 | 单题128项只发一个branch请求，所有候选完整返回；129明确拒绝 |
| 兼容 | K≤26 token级prompt不变；noul/score/概率/usage契约不变 |
| warm语义 | producer结束先于分支；不支持或关闭时沿用原路径 |
| 缓存有效 | 可观察到分支命中和实际新计算token减少；不是只看全局hit rate |
| 正确性 | GDN分支不互相污染，数值差在预先记录的容差内 |
| 性能 | 给出冷/热和DG混流的完整HTTP p50/p95及工作量；如有退化，明确受益范围并保留开关 |
| 交付诚实性 | CPU合同通过与NPU验收分别列出；无硬件时不得标记性能完成 |

本方案不预设所有场景必须加速。启用条件应依据目标部署实测收敛；若热缓存代价明显，第一阶段可保持可选开关，不用未经证明的缓存存在性启发式替代引擎事实。

## 8. 交付给审阅者的结果

建议分为可独立审阅的提交：预热回移、choice128扩展、验证与部署文档。版本规范化检查修复与文档一起交付。

本地 Codex 最终输出：

1. 修改摘要和文件清单；标出哪些是上游回移、哪些是本地扩展。
2. 原032语义合同、PERF_SOURCE合同、128边界合同结果。
3. 真实tokenizer码表验证JSON，或明确pending原因与复现命令。
4. 冷/热、不同N/K、DG混跑原始性能JSON/CSV及简短结论。
5. 目标环境启动、切换warm开关和回滚步骤。
6. 如需Ascend底层补丁，提供导致它必要的命中/状态证据，避免扩大到整套引擎升级。

## 9. 可直接复制给本地 Codex 的任务提示

> 请在现有 F00L42/vllm 的 codex/v0.30.0-qwen-decisions 工作区继续开发，先读本文件和仓库约束，保留已有修改。以当前030底座与已回移的032 Qwen Decisions API为基础，补迁官方PR #61192（ee22f682ab0d2139ae69e3cdbb762a565a166813）的shared-prefix warm机制，并实现单道choice最多128项。
>
> 先完成基线审计，再分开实现两个改动：warm使用完整已渲染prompt的LCP、先producer完成再并发原完整分支，保留salt/LoRA/priority/trace和外部usage，提供一个A/B开关；128项使用第4.1节已经实测的A–Z加102个大写双字母固定码表，不能再采用0–101数字方案。保留≤26项全部prompt与标签，只有大题改变回复指令。030已支持128个logprob_token_ids，不预先修改sampler容量，也不设置logprobs=128、不拆分choice。
>
> 更新verify_upstream.py为明确的原语义、性能来源和有意扩展合同，不删除整个模块核验。检查目标Ascend V1的真实checkpoint导出/恢复和指定token读出链；已有030能力优先复用，不先移植整套main引擎。按本文件完成CPU、真实tokenizer与可用NPU验证，提交原始证据。没有服务器时继续完成本地可完成的全部工作，将NPU数值/性能标pending_remote并给出可直接运行的验收脚本。最终说明收益、退化范围和未完成项，不把mock测试或上游AMD数据作为本地性能结果。

## 10. 固定源码与证据

- [S1 目标部署手册](https://github.com/F00L42/vllm/blob/ea2e3575d07bf49fdefa1e74ceb349896ff74f8d/docs/serving/online_serving/qwen_decisions_backport_ascend.md)
- [S2 当前main的策略实现](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/entrypoints/openai/decisions/strategies.py#L371)
- [S3 PR 61192 变更与作者实验](https://github.com/vllm-project/vllm/pull/61192)
- [S4 目标策略实现](https://github.com/F00L42/vllm/blob/ea2e3575d07bf49fdefa1e74ceb349896ff74f8d/vllm/entrypoints/generate/structured_decisions/strategies.py)
- [S5 目标Mamba同一步复用保护](https://github.com/F00L42/vllm/blob/ea2e3575d07bf49fdefa1e74ceb349896ff74f8d/vllm/v1/core/single_type_kv_cache_manager.py#L1724)
- [S6 目标Mamba checkpoint位置及有效性](https://github.com/F00L42/vllm/blob/ea2e3575d07bf49fdefa1e74ceb349896ff74f8d/vllm/v1/kv_cache_interface.py#L1073)
- [S7 目标题型与概率语义](https://github.com/F00L42/vllm/blob/ea2e3575d07bf49fdefa1e74ceb349896ff74f8d/vllm/entrypoints/generate/structured_decisions/question_types.py)
- [S8 目标指定token容量](https://github.com/F00L42/vllm/blob/ea2e3575d07bf49fdefa1e74ceb349896ff74f8d/vllm/sampling_params.py#L30)
- [S9 目标V1采样实现](https://github.com/F00L42/vllm/blob/ea2e3575d07bf49fdefa1e74ceb349896ff74f8d/vllm/v1/sample/sampler.py)
- [S10 目标服务与admission入口](https://github.com/F00L42/vllm/blob/ea2e3575d07bf49fdefa1e74ceb349896ff74f8d/vllm/entrypoints/generate/structured_decisions/serving.py)
- [S11 目标源码合同脚本](https://github.com/F00L42/vllm/blob/ea2e3575d07bf49fdefa1e74ceb349896ff74f8d/tools/decisions/verify_upstream.py)
- [S12 原v0.32.0策略](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/entrypoints/generate/structured_decisions/strategies.py)
- [S13 旧Ascend共享前缀方案](https://github.com/F00L42/vllm-ascend/blob/4d7786dbb2e15be660b73a4f63a2331bf9815d91/vllm_ascend/entrypoints/systemone/shared_prefix.py)
- [S14 Qwen3.6-27B 官方模型配置](https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/config.json)
- [S15 本次下载验证的官方 tokenizer.json](https://huggingface.co/Qwen/Qwen3.6-27B/resolve/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/tokenizer.json)
- [S16 官方 chat template](https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/chat_template.jinja)
- [S17 vLLM Qwen3.5 模型的 IsHybrid 声明](https://github.com/F00L42/vllm/blob/ea2e3575d07bf49fdefa1e74ceb349896ff74f8d/vllm/model_executor/models/qwen3_5.py#L473)

## 11. Qwen3.6-27B 码表实测与本地复验

2026-10-29 下载官方 revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` 的配置、tokenizer.json、tokenizer_config.json、vocab.json、merges.txt 与 chat_template.jinja，未下载模型权重。

| 检查 | 结果 |
| --- | --- |
| 128 个标签孤立分词 | 全部单 token |
| 128 个 token ID | 全部不同，均非 special/added token |
| 官方关闭 thinking 的回复尾部 | 双换行，对应 `[271]` |
| 在回复尾部追加标签 | 保留原 tail，只追加一个指定 token |
| 完整 prompt 后追加标签 | 原 prompt IDs 完全保持，只追加一个指定 token |
| 测试上下文 | 4 种 state × 有/无 system × 26/27/128 项，共 24 种 |
| 上下文与标签组合 | 24 × 128 = 3072，全部通过 |
| AutoTokenizer 与原始 tokenizers.Tokenizer | 全部一致 |
| 原数字候选方案 | 92 项失败，已更换 |

环境：Transformers 5.5.4、tokenizers 0.22.2、Jinja2 3.1.6，加载类为 TokenizersBackend。此验证覆盖分词与指定回复起点，不代表模型在128项业务任务上的准确率或NPU性能已经验收。

| 文件 | SHA256 |
| --- | --- |
| tokenizer.json | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| tokenizer_config.json | `5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b` |
| chat_template.jinja | `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259` |

`qwen36_choice128_verified.json` 含完整 labels、token_ids、label_to_token_id、文件哈希和每个测试上下文记录。`verify_qwen36_choice128.py` 包含固定码表，可在本地部署环境不加载权重地复验：

```bash
python verify_qwen36_choice128.py \
  --tokenizer /实际模型目录/Qwen3.6-27B \
  --output qwen36_choice128_local_verified.json
```

如服务指定了外部模板，加 `--chat-template /实际模板路径`。脚本有意严格要求与本次官方模板相同的双换行回复边界；其他有效但不同的回复起点会明确失败，需要重新验证并记录自己的边界，不能直接跳过检查。实际请求参数改变 generation prompt 时，也必须继续保留服务端的 tail 一致性检查。
