# 在 Ascend 0.30 容器部署官方 Qwen Decisions 回移版

此分支以官方 vLLM `v0.30.0` 为底座，回移官方 `v0.32.0` 的 Qwen
`NextTokenStrategy` 和 `/v1/systemone`。在配套 Ascend 0.30 容器里，只升级
vLLM Python 包，保留镜像提供的 vllm-ascend、torch、torch-npu、Triton 和 CANN。

本地已验证源码语义和离线服务合同。真实容器安装、NPU 推理、混跑、图执行、缓存命中、
数值对比和性能均为 `pending_remote`。本手册提供这些步骤，不能用本地测试替代上机验收。

## 迁移来源和范围

| 项目 | 固定基线 |
| --- | --- |
| 代码仓库 | [F00L42/vllm](https://github.com/F00L42/vllm) |
| 分支 | `codex/v0.30.0-qwen-decisions` |
| vLLM 底座 | `v0.30.0`，`ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| Decisions 来源 | `v0.32.0`，`5df63a7203cdb9328d824a27b682e77c7b9928be` |
| Ascend 配套源码 | `v0.30.0rc2`，`8fde6d4e335fe300e2b949aab0e31659cce0811d` |
| A2 镜像 | `quay.io/ascend/vllm-ascend:v0.30.0rc2` |
| A2 镜像 digest | `sha256:6348403a0ebb19f52bc1b7497de1daeb6df0e4650f9051c169e47a011527a6c3` |
| A3 镜像 | `quay.io/ascend/vllm-ascend:v0.30.0rc2-a3` |
| A3 镜像 digest | `sha256:0676d2dbdd480c2c456aacbeee486681c7c33a81aa83faa1edc9168a0f4f7fb6` |

镜像 tag 和 digest 在 2026-10-20 查询。使用 rc2；Ascend 官方说明 rc1 的 CANN
基础镜像构建存在问题，需要升级。不要把上游 vLLM `v0.30.0` tag 和 Ascend `v0.30.0rc2`
tag 混为同一个仓库的版本。

来源：
[vLLM v0.32.0](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)、
[官方读出策略](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/entrypoints/generate/structured_decisions/strategies.py)、
[Ascend rc2](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.30.0rc2)。

保留官方 tag 注册的这三个架构，按 checkpoint `config.json` 的 `architectures` 判断：

- `Qwen3ForCausalLM`
- `Qwen3_5ForConditionalGeneration`
- `Qwen3_5MoeForConditionalGeneration`

这份名单描述 API 策略选择；模型本身能否在 Ascend rc2 加载，仍受镜像支持范围限制。
例如该 tag 未注册 `Qwen3MoeForCausalLM`。不因商品名称含 Qwen 就自动放行。
此路径只读文本，Qwen3.5 的 Decisions 请求也不接受 image、audio、video。
其他官方策略及模型不在本次回移范围。

保留官方26项以内的 prompt、真实回复起点的单 token 校验、每题一次
`generate(max_tokens=1, temperature=0, logprob_token_ids=...)`、并发聚合、
`choice/noul/score`、`label_mass`、`argmax_is_label`、LoRA/priority/cache_salt
传递和 usage 计数。

兼容改动只有：将 chat-utils 导入改回 0.30 路径；在本服务类中补入官方的 trace-header
方法；在 0.30 的 app-state 和 router 初始化处接入服务。未回移 0.32 的模型、引擎、
调度器、采样器、依赖升级或非 Qwen 策略。后续新增的预热与128项扩展见下一节。

旧分支 `codex/systemone-token-readout` 的 `ascend_systemone` endpoint、自定义 prompt、
温度配置、共享前缀预填屏障和 APC 实验补丁不属于本次基线。官方方式依赖原生 APC 和调度器；
完整提交 prompt 不代表每题一定重算，也不保证冷缓存时公共 state 只算一次。

## 本次补迁：预热与128项扩展

预热来自原始 v0.32.0 tag 之后的官方 PR #61192，固定提交
`ee22f682ab0d2139ae69e3cdbb762a565a166813`，不是原032发布包已有的功能。
服务级开关 `VLLM_DECISIONS_WARM_SHARED_PREFIX` 默认 `0`，用于 A/B 和回滚。

启用时设置 `export VLLM_DECISIONS_WARM_SHARED_PREFIX=1` 并完整重启服务。
多题、hybrid、APC开启且完整渲染prompt的公共前缀至少128tokens时，先完成一个
`-prefix` 内部请求，再并发提交各题原始完整prompt。额外生成的token丢弃，外部usage仅统计各题。
当前先支持DP=1、无外部KVconnector的缓存域；多DP、外部connector和媒体输入跳过预热。
空流、error/abort或未完成producer会明确失败，不继续分支，也不吞错重试。

预热长度不是实际命中长度。030的Qwen GDN没有声明任意内部prefill checkpoint导出，
实际复用仍取决于已有最终状态、块对齐、状态恢复与淘汰。没有NPU命中与数值结果前，
不保证长state一定只计算一次。详细核查和实验步骤见
[迁移与验收记录](../../design/decisions_030_extensions.md)。

128项是本地扩展：固定使用A–Z和102个已验证的大写双字母。≤26项的prompt逐字保持原样，
>26项才改为要求精确标签；noul/score和概率、confidence契约保持不变。
启动时验证完整码表，不兼容的tokenizer/template以ERROR日志说明并禁用该endpoint，不隐式截断。
官方Qwen3.6-27B固定revision已本地复验；其他模型和部署自定义模板需要单独复验。

## 创建容器

下面示例为 A2，使用已固定的镜像 digest。按实际硬件调整 NPU 设备和模型目录。
四卡/多卡时需把所用 `/dev/davinciN` 都映射进去。以下命令在宿主机执行：

```bash
NPU_SMI_PATH="$(command -v npu-smi)"
test -n "$NPU_SMI_PATH"
docker run -it --name qwen-decisions-030 \
  --network host --shm-size 16g \
  --device /dev/davinci0 --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$NPU_SMI_PATH:/usr/local/bin/npu-smi:ro" \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /models:/models:ro -v /workspace:/workspace \
  quay.io/ascend/vllm-ascend@sha256:6348403a0ebb19f52bc1b7497de1daeb6df0e4650f9051c169e47a011527a6c3 \
  bash
```

A3 使用上表 A3 digest，一张卡需要两个 DIE：在容器命令里增加
`--device /dev/davinci1`，并在后续启动命令中设置 `ASCEND_RT_VISIBLE_DEVICES=0,1`
和 `--tensor-parallel-size 2`。实际节点编号按宿主机 `npu-smi info` 调整；更多设备组合见
[官方 A3 容器说明](https://github.com/vllm-project/vllm-ascend/blob/8fde6d4e335fe300e2b949aab0e31659cce0811d/docs/source/getting_started/quick_start/ascend_image/atlas-a3.inc.md)。

容器内先保存环境清单。rc2 构建会装入 snapshot 依赖，因此以实际清单为准，不按源码
`requirements.txt` 猜测镜像内部版本。

```bash
mkdir -p /workspace/decisions-evidence
python -m pip freeze > /workspace/decisions-evidence/before-freeze.txt
python - <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
packages = ('vllm', 'vllm-ascend', 'torch', 'torch-npu', 'triton-ascend', 'triton', 'transformers')
versions = {}
for name in packages:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
text = json.dumps(versions, indent=2)
print(text)
Path('/workspace/decisions-evidence/before-versions.json').write_text(text + '\n')
PY
npu-smi info
python - <<'PY'
import torch
import torch_npu
import vllm
import vllm_ascend
assert torch.npu.is_available()
print('vLLM Ascend environment: OK')
PY
```

## 只安装定制 vLLM

在容器已有的 `/vllm-workspace/vllm` 仓库里拉取定制分支，然后使用原解释器重新安装。
先确认 `git status --short` 没有待保留的本地修改；下面的切换和拉取不会强制覆盖本地代码。

```bash
set -e
cd /vllm-workspace/vllm
git status --short
git fetch https://github.com/F00L42/vllm.git codex/v0.30.0-qwen-decisions
if git show-ref --verify --quiet refs/heads/codex/v0.30.0-qwen-decisions; then
  git switch codex/v0.30.0-qwen-decisions
else
  git switch -c codex/v0.30.0-qwen-decisions FETCH_HEAD
fi
git pull --ff-only https://github.com/F00L42/vllm.git codex/v0.30.0-qwen-decisions

VLLM_TARGET_DEVICE=empty \
VLLM_VERSION_OVERRIDE=0.30.0+qwen.decisions.032 \
python -m pip install --no-deps --no-build-isolation -e .

python - <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
import vllm
from vllm.entrypoints.generate.structured_decisions.strategies import READ_STRATEGIES
before = json.loads(Path('/workspace/decisions-evidence/before-versions.json').read_text())
for name, version in before.items():
    if name != 'vllm':
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        assert installed == version, (name, version, installed)
from packaging.version import Version
assert Version(metadata.version('vllm')) == Version('0.30.0+qwen.decisions.032')
print(vllm.__version__, vllm.__file__)
print(sorted(READ_STRATEGIES))
PY
python -m pip freeze > /workspace/decisions-evidence/after-freeze.txt
```

`--no-deps --no-build-isolation` 保留镜像的 NPU 软件栈；上游 0.30 的隔离构建要求与
Ascend 镜像里的 torch 配套不同。`VLLM_TARGET_DEVICE=empty` 沿用 Ascend 的纯 Python
vLLM 安装方式；`VLLM_VERSION_OVERRIDE` 固定 0.30 包版本，避免分支提交被推算成下一版本。
这些都是已有构建参数，不新增运行时环境开关。

保留 0.30 公共服务行为：未知模型仍返回 404，其错误文本不包含 0.32 新增的别名提示；
请求体日志也沿用 0.30。这里的官方一致性指 Qwen 请求、读出与响应语义，不承诺所有公共
服务错误文本和日志逐字一致。

必须在同一个解释器里安装和启动；安装后完整重启 API server 和所有 workers。
`vllm.__file__` 应指向 `/vllm-workspace/vllm/vllm/`。后续更新也在此目录执行上述拉取和安装步骤，
拉取使用显式 fork 地址，保留已有 `origin` 设置。此回移不需要安装旧版 Ascend 自定义代码，也不需要
`--additional-config '{"systemone_config":...}'`。

## 启动服务

先用镜像已支持的 checkpoint 做验收。以下为单卡起始配置，容量和 TP 数按实际模型调整。
图执行和异步调度使用框架原生选择；仅在排障时临时添加
`--enforce-eager --no-async-scheduling`。新功能不强制这些参数，比较性能时双方保持相同执行配置。

```bash
unset VLLM_PLUGINS
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_DECISIONS_WARM_SHARED_PREFIX=0
export ASCEND_RT_VISIBLE_DEVICES=0

vllm serve /models/qwen-checkpoint \
  --served-model-name qwen-decisions \
  --runner generate --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 8192 --max-num-batched-tokens 2048 --max-num-seqs 8 \
  --enable-prefix-caching --enable-chunked-prefill \
  --logprobs-mode raw_logprobs \
  --host 0.0.0.0 --port 8010
```

`unset VLLM_PLUGINS` 让标准插件发现机制加载镜像的 Ascend 插件，并移除旧
`ascend_systemone` 白名单。若部署依赖其他插件，可显式配置标准 Ascend 插件列表，但不能
加载旧 `ascend_systemone`，避免两个同名路由。

**固定 V1 runner 是必要条件。** rc2 的 V1 `AscendSampler` 继承上游候选 token
logprobs 路径；V2 的 `compute_topk_logprobs` 明确未支持自定义 `logprob_token_ids`。
使用 `raw_logprobs` 保持 `label_mass` 的全词表概率含义；不使用 logits 模式。
128 个 choice 标签不要求把 `--max-logprobs` 提到 128，0.30 对指定 token IDs 使用单独校验。

对于 Qwen3.5/GDN 混合模型，先核对 checkpoint 架构和镜像支持，再按其混合缓存要求增加
`--mamba-cache-mode align`；不要把普通 Qwen3 的缓存验收直接视为 GDN 状态复用证据。

## 请求与验收

```bash
curl http://127.0.0.1:8010/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen-decisions",
    "state": "My card was charged twice. Please refund the extra charge.",
    "questions": {
      "team": {
        "type": "choice",
        "instructions": "Which team handles this ticket?",
        "criteria": {"billing": "Payments", "shipping": "Delivery"}
      },
      "money": {"type": "noul", "instructions": "Does this concern money?"},
      "urgency": {
        "type": "score",
        "instructions": "How urgent is the ticket?",
        "criteria": ["low", "medium", "high"]
      }
    }
  }'
```

| 官方语义 | 本分支行为 |
| --- | --- |
| 每次最多问题数 | 64 |
| choice | 字典 criteria，1–128 项；前26个标签仍为 `A…Z` |
| noul | `yes/no` 标签；`noul` 是 yes 的候选归一化概率 |
| score | criteria 为有序列表，2–10 级；score 为从 0 开始的期望等级 |
| probabilities | 仅在候选标签内归一化 |
| confidence | 最大候选归一化概率 × `label_mass` |
| diagnostics | 每题 `label_mass` 和 `argmax_is_label` |
| thinking | 默认关闭；显式开启报错 |
| usage | 各题完整输入 token 数相加，输出通常每题 1 token；不表示实际新算 token 数 |

旧请求中的 choice 列表、score 字典、旧温度和旧概率定义需按以上协议修改。
prompt 和标签已改为官方版本，同一权重也不保证与旧补丁返回相同概率。

运行真实服务检查并保存原始响应，失败时仍会写出诊断文件：

```bash
cd /vllm-workspace/vllm
python tools/decisions/smoke_qwen.py \
  --base-url http://127.0.0.1:8010 --model qwen-decisions \
  --output /workspace/decisions-evidence/smoke.json
```

检查包含三种题型、26/128 项 choice、129 项拒绝、10 级 score、单选项 confidence、输入校验、thinking/媒体拒绝，以及并发 Decision
和 Chat Generation。通过只证明这些请求的服务行为，不证明质量、压力容量或性能。

若另有原版 vLLM `v0.32.0` 服务，加载完全相同的权重、tokenizer、chat template、dtype
和相应配置，可加上参考端点进行数值对比：

```bash
python tools/decisions/smoke_qwen.py \
  --base-url http://127.0.0.1:8010 --model qwen-decisions \
  --reference-url http://reference-host:8010 --reference-model qwen-decisions \
  --atol 0.0001 --output /workspace/decisions-evidence/compare-032.json
```

对比 `answers`、`diagnostics` 和 usage，忽略动态 ID、时间和模型别名。
跨设备数值可有差异；容差是验收条件，应在实验前固定。该脚本只有一个合成输入，不能替代
业务回放、长上下文和冷/热缓存实验。性能基线还需分别记录 TTFT、完成 questions/s、
排队、P50/P95/P99、APC 命中和 DG 混跑对生成延迟的影响。

## 本地检查与交付边界

完整 Git 仓库可运行静态语义比对，需要本地保留上述两个 tag：

```bash
uv run --no-project --python .venv/bin/python python tools/decisions/verify_upstream.py
```

该检查对比官方核心模块、保留的策略和服务方法的 AST，允许已说明的导入路径变更、
Qwen 注册范围裁剪和 trace 方法搬移。离线测试使用真实 0.30 的协议和部分渲染/采样代码，
推理引擎和 tokenizer 使用替身；具体命令和边界见
[离线验证说明](../../../tools/decisions/README.md)。

回滚最直接的方法是停止并移除本次容器，用相同 digest 重新创建；模型和证据目录位于
宿主机挂载目录。保留旧容器时先停服务，在新容器完成验收后再切流量。
