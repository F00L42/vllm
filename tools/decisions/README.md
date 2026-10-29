# Qwen Decisions 回移验证

部署步骤和固定的源码、镜像基线见
[Ascend 部署手册](../../docs/serving/online_serving/qwen_decisions_backport_ascend.md)。
以下命令在本仓库根目录执行。离线测试不会安装或启动完整 vLLM。

## Windows 离线合同测试

使用独立环境，避免修改部署容器的软件栈：

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe pytest pytest-asyncio numpy msgspec pydantic 'fastapi>=0.133.0,<0.137.0' httpx pillow regex
uv run --no-project --python .venv/Scripts/python.exe python tools/decisions/run_offline_tests.py -q
uv run --no-project --python .venv/Scripts/python.exe python tools/decisions/verify_upstream.py
```

已有 `.venv` 时跳过创建。Linux 的解释器路径改为 `.venv/bin/python`。
`verify_upstream.py` 需要完整 Git 仓库及 `v0.30.0`、`v0.32.0` 两个 tag；
缺少 tag 时，在仓库中执行 `git fetch https://github.com/vllm-project/vllm.git tag v0.30.0 tag v0.32.0`。
PERF_SOURCE另需 `git fetch https://github.com/vllm-project/vllm.git ee22f682ab0d2139ae69e3cdbb762a565a166813`。
容器内拉取定制分支和安装均在 `/vllm-workspace/vllm` 下进行，命令见部署手册。

本次离线结果：**153 项通过**；原始 `ea2e357` 不可变快照的62项亦重新通过。
覆盖原题型、Qwen策略、0.30服务接入及新增预热和128项合同：

- `choice/noul/score` 的概率、`label_mass`、confidence、score legend 和 usage。
- 单选项的 confidence、全词表 argmax 不在候选内、标签 token 校验。
- 官方三个 Qwen 架构、任务及 logprobs 模式限制、thinking 和媒体拒绝。
- 每题并发执行、返回顺序、LoRA、priority、cache salt、trace header 和请求 ID。
- 请求错误、缺失引擎输出或 logprobs，以及渲染失败时不提交部分请求。
- FastAPI 路由响应和服务初始化函数的支持/拒绝路径。

### 离线结果的边界

测试完整导入本次回移的模块，运行真实 0.30 的 `SamplingParams`、
`ChatParams/TokenizeParams`、输入结构、输出对象、`BaseServing`、异步合并函数、
W3C trace 提取函数、FastAPI 路由和取消/负载装饰器。
为绕开 Windows 上的完整推理运行时导入，runner 通过 AST 提取并执行原始
`OnlineRenderer.preprocess_chat` 方法及少量依赖定义，提供显式的导入边界替身。

**tokenizer、低层 renderer 和 engine 使用替身**；模型配置、媒体连接器等重型导入
使用占位对象。没有加载模型权重，也没有运行 NPU 算子。
这不能证明实际 tokenizer、完整 server 启动/异常注册、插件发现、模型加载、
APC 命中、GDN 状态复用、DG 混跑、图执行、数值精度或性能。
静态 AST 比对也不能证明这些运行时行为。

## 完整运行时和真实 tokenizer 测试

在已按部署手册安装本分支的 Linux 环境中，使用同一个解释器运行：

```bash
python -m pytest -q \
  tests/entrypoints/generate/structured_decisions/test_question_types.py \
  tests/entrypoints/generate/structured_decisions/test_next_token_strategy.py \
  tests/entrypoints/generate/structured_decisions/test_next_token_tokenizer.py
```

这里不使用离线 runner，测试会正常导入真实 vLLM 模块。环境需有项目测试依赖和
`Qwen/Qwen3-0.6B` tokenizer 文件；首次运行可能下载 tokenizer。
`test_next_token_tokenizer.py` 从官方策略测试提取 Qwen 用例，校验实际回复位置的
`A…Z`、`yes/no`、数字标签和冒号后的 BPE 合并拒绝行为，不需要加载模型权重。

## 真实模型端到端测试

`test_structured_decisions_e2e.py` 基于官方 `v0.32.0`，仅将选项边界扩展到128/129并增加输入长度预算，会自行启动
`Qwen/Qwen3-0.6B` 服务并加载权重，需要完整测试依赖、模型文件和可用设备。
Ascend 环境需保持部署手册要求的 V1 runner：

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
python -m pytest -q \
  tests/entrypoints/generate/structured_decisions/test_structured_decisions_e2e.py
```

也可以检查部署手册启动的现有服务；此客户端只依赖 Python 标准库：

```bash
python tools/decisions/smoke_qwen.py \
  --base-url http://127.0.0.1:8010 --model qwen-decisions \
  --output /workspace/decisions-evidence/smoke.json
```

与原版 0.32 服务对比时，两边需使用相同权重、tokenizer、chat template、dtype 和
执行参数；先固定验收容差，再执行：

```bash
python tools/decisions/smoke_qwen.py \
  --base-url http://127.0.0.1:8010 --model qwen-decisions \
  --reference-url http://reference-host:8010 --reference-model qwen-decisions \
  --atol 0.0001 --output /workspace/decisions-evidence/compare-032.json
```

客户端保存原始响应和检查结果。通过只证明所覆盖请求的服务行为与合成输入对比，
不能替代业务数据回放或性能基准。上述完整运行时、真实 tokenizer、模型 E2E 和远端
数值/性能验证在本次 Windows 交付中均为 `pending_remote`。

## 预热、128项及真实CPU gather

源码合同的原始语义、PERF_SOURCE及有意扩展分别核验；不能通过删除整个策略检查放宽合同。
完整本地证据、缓存能力审计及三组远端实验见
[迁移与验收记录](../../docs/design/decisions_030_extensions.md)。

在已有Transformers、tokenizers与Jinja2的环境，只加载tokenizer复验：

```bash
python tools/decisions/verify_qwen36_choice128.py \
  --tokenizer /models/Qwen3.6-27B \
  --output /workspace/decisions-evidence/tokenizer-128.json
python tools/decisions/probe_sampler_cpu.py \
  --output /workspace/decisions-evidence/sampler-cpu.json
```

CPU probe需要有torch的解释器，或传入现有CPU包目录`--torch-site-packages`。
其结果只覆盖原030 gather数学、异长候选、padding和索引重排，不证明NPU buffer/runner。
真实服务实验使用`benchmark_qwen.py`的warm/gather/business三种模式，详见上述记录。

本机应用程序控制阻止旧venv的Python launcher；实际离线和分词检查使用已获准的
uv基础解释器加载该venv依赖，没有改变系统控制策略。此环境问题不代表目标容器行为。
