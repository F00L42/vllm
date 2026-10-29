# 030 Decisions：预热回移与128项扩展

2026-10-29。本次在 `codex/v0.30.0-qwen-decisions`、原提交 `ea2e357` 上继续开发。
这是两项可分别验证的改变：预热开关只改变多题的执行顺序；128项扩展只改变大题的标签和指令。

## 来源与实现

- vLLM底座：`ced6857afa0ea7b2e3f0846a62e1394e90f15607`（v0.30.0）。
- 原API：`5df63a7203cdb9328d824a27b682e77c7b9928be`（v0.32.0）。
- 预热：官方[PR #61192](https://github.com/vllm-project/vllm/pull/61192)，
  `ee22f682ab0d2139ae69e3cdbb762a565a166813`。原v0.32.0 tag没有此优化。
- 部署：Ascend v0.30.0rc2，`8fde6d4e335fe300e2b949aab0e31659cce0811d`，V1 runner。
- 输入任务书：[原设计](SystemOne_030_WarmPrefix_128Choice_Codex.md)。其中上游AMD性能数字是参考，
  不是本次Ascend实测；代码和下面的验证记录为本地交付结果。

预热对所有完整渲染后的prompt求LCP，正常消费一个无候选logprobs的`-prefix`请求后，
才提交各题原始完整prompt。producer生成的token丢弃，外部usage仍只累计分支。
保留上游多题、hybrid、APC、LCP至少128tokens四个条件，不手动裁到物理块。

本地增加的差异有：默认关闭的`VLLM_DECISIONS_WARM_SHARED_PREFIX`开关；空输入LCP为0；
producer空流、未完成、error/abort及异常处理；按原AsyncLLM生命周期关闭生成器；
DP>1、外部KVconnector和媒体的跳过条件；带父请求ID的DEBUG聚合日志。
失败不会吞错后重试，也不会提交分支。没有缓存存在性集合、跨HTTP合并或缓存固定引用。

choice使用A–Z加102个固定大写双字母，逐项与提供的码表一致。≤26项标签及prompt逐字不变，
>26项才使用精确标签指令；noul仍是yes/no，score仍为0–9。单题128项仅有一个分支请求，
不会因为选项多而预热，也不设置`logprobs=128`或候选mask。

启动时按实际tokenizer和模型词表验证全部140个题型标签：choice128、noul2、score10。
拒绝拆分、重复、special/added和越界ID。失败时记录模型、tokenizer、模板哈希及原因，
禁用该endpoint，不隐式降级。`label_reads.py`另拒绝NaN、正无穷和全负无穷分数，
允许有限分数与负无穷共存（零概率），避免产生NaN响应。

## 030与Ascend缓存能力审计

**预热成功不等于LCP全部命中。** 030的通用Mamba checkpoint能力不能直接当作Qwen能力。

| 链路 | 核查结果 |
| --- | --- |
| 030 `layers/mamba/abstract.py` 的MambaBase | 沿用MambaSpec默认值：内部checkpoint数量0、alignment为None |
| 030 `single_type_kv_cache_manager.py` | 保留`cached_blocks_this_step`保护，同一步未完成状态不能交给分支 |
| rc2 `patch_mamba_manager.py:49` | 调用上游分配逻辑，未绕过该保护 |
| rc2 `patch_mamba_config.py:159` | APC/align下设置Mamba block size与Attention block size对应 |
| rc2 `gdn_attn_builder.py:649`、`:848` | 从block table/computed tokens确定状态索引和初始状态 |
| rc2 `gdn.py:573` | Prefill读取初始状态并写最终recurrent state，未发现任意内部checkpoint导出 |
| rc2 `patch_mamba_utils.py:384` | 按已计算tokens及块位置恢复、复制缓存状态到工作块 |

以上Ascend路径固定在
[rc2源码](https://github.com/vllm-project/vllm-ascend/tree/8fde6d4e335fe300e2b949aab0e31659cce0811d/vllm_ascend)。
已有对齐/最终/partial-tail状态复用可供该编排使用，但实际每个分支命中长度、状态内容正确性、
淘汰或抢占后的恢复仍需NPU证据。本次没有据静态审计增加Ascend补丁，也没有改scheduler、
KV manager、sampler容量、模型或算子。

## 已运行的本地验证

| 验证 | 结果与边界 |
| --- | --- |
| 原提交不可变快照 | 62项离线测试通过 |
| 新合同回归 | 153项通过；engine/tokenizer替身用于服务层测试 |
| 真实Qwen3.6 tokenizer | 24上下文×128标签=3072组通过；没有加载权重 |
| 实际策略初始化 | 128个choice ID与参考一致；联合映射140个；1–26项完整prompt IDs与基线一致 |
| 真实CPU sampler | 原030 gather/输出结构/rank helper在torch2.10 CPU运行通过 |
| 源码合同 | 原032未改语义、固定PERF_SOURCE算法与producer参数、有意扩展分别核验 |

真实tokenizer固定在`Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`。
本地复验使用Transformers5.10.4、tokenizers0.22.2和Jinja2 3.1.6；四个核心文件哈希与
输入参考一致。参考报告使用的Transformers5.5.4不是本次执行版本。

证据：

- [输入参考码表](qwen36_choice128_verified.json)
- [本地真实tokenizer复验](qwen36_choice128_local_verified.json)
- [生产策略与真实tokenizer对照](qwen36_choice128_service_tokenizer.json)
- [真实CPU gather结果](decisions_030_sampler_cpu.json)

CPU sampler覆盖2/26/128异长候选、129输出列、padding为负无穷、sampled token在集合内外、
索引重排、空选择；使用固定030源码，不替换数学运算，未跳过annotation。
它不验证NPU buffer、设备拷贝或Ascend runner集成。

源码合同通过`extensions.json`精确登记修改的符号及AST哈希；其他声明仍与原回移基线一致。
概率公式、旧题型、HTTP协议和usage未放宽核验；LCP算法、阈值和producer输入/参数另与
PERF_SOURCE比较。将来调整代码需要审阅并同步这些有意差异，不能删除整个模块检查。

## 启动与A/B

按照[容器部署手册](../serving/online_serving/qwen_decisions_backport_ascend.md)
在`/vllm-workspace/vllm`拉取、安装。继续使用`--no-deps --no-build-isolation`、V1 runner
和`raw_logprobs`。安装版本用`packaging.version.Version`比较，接受PEP440将`.032`规范化为`.32`。

预热关闭：`export VLLM_DECISIONS_WARM_SHARED_PREFIX=0`；开启改为`1`，每次完整重启服务。
仅回滚预热用`0`即可，128项能力不受影响。恢复26项旧行为则切回`ea2e357`并重新安装、重启。

常规启动不强制eager或关闭async；排障可以临时增加它们，但性能对照两边必须使用相同设置。
先验收DP1、无外部KVconnector，再考虑扩大缓存域。

## 远端验收命令

在同一硬件上分别运行warm关闭/开启服务，示例端口8010/8011。
并列实例必须有相同硬件资源或分时运行，避免把设备竞争当作warm收益。
先完成算子和图编译预热，再测试逻辑冷KV。`state.txt`由操作者准备，并记录实际渲染token数；
脚本不会把字符数冒充token数。

```bash
cd /vllm-workspace/vllm
python tools/decisions/smoke_qwen.py --base-url http://127.0.0.1:8010 \
  --model qwen-decisions --output /workspace/decisions-evidence/smoke-128.json
python tools/decisions/benchmark_qwen.py --mode warm \
  --off-url http://127.0.0.1:8010 --on-url http://127.0.0.1:8011 \
  --model qwen-decisions --state-file /workspace/state.txt \
  --questions 1 4 16 --choices 26 128 --repeats 5 \
  --output /workspace/decisions-evidence/warm.json
```

记录实际vLLM/Ascend SHA、模型/tokenizer/template哈希、TP/DP、chunk budget、缓存和图/调度配置，
作为`--off-metadata`、`--on-metadata`的JSON输入。脚本将其标为操作者提供信息，
不冒充已从服务端验证配置。

固定prompt测候选读出成本时，`reply-start.txt`必须是同一个完整渲染prompt，码表必须属于部署模型：

```bash
python tools/decisions/benchmark_qwen.py --mode gather \
  --url http://127.0.0.1:8010 --model qwen-decisions \
  --prompt-file /workspace/reply-start.txt \
  --codebook docs/design/qwen36_choice128_local_verified.json \
  --output /workspace/decisions-evidence/gather.json
python tools/decisions/benchmark_qwen.py --mode business \
  --url http://127.0.0.1:8010 --model qwen-decisions \
  --state-file /workspace/state.txt \
  --output /workspace/decisions-evidence/business.json
```

gather模式只改变26/64/128个指定ID，不改变prompt，也不用于不同K的答案质量比较。
business模式才增加实际选项文本。warm模式每次逻辑冷请求用新salt；热请求先消费相同salt
的prime，再记录重复请求。脚本保存prime及所有原始响应。

加入`--generation-tokens 64`可同时发出流式Generation，记录客户端TTFT和TPOT估计。
这不是完整负载发生器，也不等价于服务端逐token ITL。并发父请求可用多个客户端运行，
需另外统计整个测量窗口的吞吐。

`VLLM_LOGGING_LEVEL=DEBUG`时可按`decision-<X-Request-Id>`关联warm与分支聚合日志。
正常性能测量应关闭DEBUG并使用独立观测轮次，避免日志干扰。
HTTP协议没有新增私有缓存/调度指标，因此脚本对它们保留null；每步新计算tokens、抢占、
checkpoint alignment和分支恢复正确性需要服务端观测，不能由usage反推。

真实容器安装、NPU数值、冷/热缓存命中、抢占/淘汰后的GDN状态恢复、图/async组合、
业务128项准确率和性能均为`pending_remote`。没有运行目标NPU实验，没有引用上游AMD倍数
作为本地收益；也没有宣称128项总延迟不变。
