# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen backport contracts with a fake tokenizer and fake engine.

Unlike model E2E tests, these exercise service semantics and v0.30 renderer
interfaces without downloading a model or executing an accelerator kernel.
"""

import asyncio
import math
from argparse import Namespace
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.datastructures import State

from vllm.entrypoints.generate.structured_decisions.api_router import (
    init_structured_decisions_state,
    register_structured_decisions_api_router,
)
from vllm.entrypoints.generate.structured_decisions.protocol import (
    ReadPromptRequest,
    StructuredDecisionRequest,
    StructuredDecisionResponse,
)
from vllm.entrypoints.generate.structured_decisions.question_types import (
    LABELS,
)
from vllm.entrypoints.generate.structured_decisions.serving import (
    ServingStructuredDecisions,
)
from vllm.entrypoints.generate.structured_decisions.strategies import (
    READ_STRATEGIES,
    NextTokenStrategy,
    ReadContext,
    reply_label_ids,
    select_read_strategy,
)
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.exceptions import VLLMValidationError
from vllm.inputs import tokens_input
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels
    from vllm.renderers import BaseRenderer
    from vllm.tokenizers import TokenizerLike

MODEL = "test-qwen"
ARCHITECTURES = (
    "Qwen3ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
)
TRACE = {"traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"}


class Tokenizer:
    """Deterministic stand-in; real Qwen tokenization belongs to model E2E."""

    max_chars_per_token = 4

    def get_added_vocab(self):
        return {"assistant": 10000}

    def encode(self, text, **kwargs):
        if text.endswith("yes"):
            return [*map(ord, text[:-3]), 10001]
        if text.endswith("no"):
            return [*map(ord, text[:-2]), 10002]
        return list(map(ord, text))

    def decode(self, tokens):
        special = {10000: "<assistant>", 10001: "yes", 10002: "no"}
        return "".join(special.get(t, chr(t)) for t in tokens)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs.get("enable_thinking") is False
        text = "\n".join(m["role"] + ": " + m["content"] for m in messages)
        return self.encode(text) + [10000, 10, 10]


class Renderer:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.calls = []
        self.bad_tail = False

    def get_tokenizer(self):
        return self.tokenizer

    async def render_chat_async(
        self, messages, chat_params, tok_params, *, prompt_extras, skip_mm_cache
    ):
        self.calls.append(
            (messages, chat_params, tok_params, prompt_extras, skip_mm_cache)
        )
        ids = self.tokenizer.apply_chat_template(
            messages[0], **chat_params.get_apply_chat_template_kwargs()
        )
        tok_params._token_len_check(self.tokenizer, ids)
        if self.bad_tail:
            ids.append(0)
        return [messages[0]], [
            tokens_input(ids, cache_salt=prompt_extras.get("cache_salt"))
        ]


class Engine:
    def __init__(self, architecture=ARCHITECTURES[0]):
        self.model_config = SimpleNamespace(
            architecture=architecture,
            logprobs_mode="raw_logprobs",
            model=MODEL,
            max_model_len=4096,
            is_encoder_decoder=False,
            multimodal_config=None,
            enable_prompt_embeds=False,
        )
        self.errored = False
        self.calls = []
        self.admissions = []
        self.failure = None
        self.selected_token = 99999

    def check_admission(self, count):
        self.admissions.append(count)

    async def generate(self, prompt, params, request_id, **metadata):
        self.calls.append((prompt, params, request_id, metadata))
        await asyncio.sleep(0)
        if self.failure == "empty_stream":
            return
        outputs = []
        if self.failure != "empty_outputs":
            values = [0.04, 0.16, 0.10]
            logprobs = {
                token: Logprob(math.log(values[index]))
                for index, token in enumerate(params.logprob_token_ids)
            }
            if self.failure == "missing_label":
                logprobs.pop(params.logprob_token_ids[-1])
            outputs = [
                CompletionOutput(
                    index=0,
                    text="outside labels",
                    token_ids=[self.selected_token],
                    cumulative_logprob=None,
                    logprobs=None if self.failure == "no_logprobs" else [logprobs],
                    finish_reason="error"
                    if self.failure == "engine_error"
                    else "length",
                )
            ]
        yield RequestOutput(
            request_id=request_id,
            prompt=None,
            prompt_token_ids=prompt["prompt_token_ids"],
            prompt_logprobs=None,
            outputs=outputs,
            finished=True,
        )


class Models:
    def __init__(self):
        self.lora_requests = {}

    def is_base_model(self, name):
        return name == MODEL

    def model_name(self, adapter):
        return MODEL if adapter is None else adapter.lora_name


@pytest.fixture
def setup_service():
    engine = Engine()
    renderer = Renderer()
    online = OnlineRenderer.__new__(OnlineRenderer)
    online.model_config = cast("ModelConfig", engine.model_config)
    online.renderer = cast("BaseRenderer", renderer)
    strategy = NextTokenStrategy(
        ReadContext(cast("EngineClient", engine), online, None, "auto", {})
    )
    models = cast("OpenAIServingModels", Models())
    return ServingStructuredDecisions(models, strategy), engine, renderer


def request(**overrides):
    return StructuredDecisionRequest(
        **(
            {
                "model": MODEL,
                "state": {"message": "Français 日本語"},
                "questions": {
                    "team": {
                        "type": "choice",
                        "instructions": "Which team?",
                        "criteria": {"billing": "money", "shipping": None},
                    },
                },
                "request_id": "fixed",
            }
            | overrides
        )
    )


@pytest.mark.asyncio
async def test_choice_noul_score_use_candidate_probabilities_and_full_vocab_confidence(
    setup_service,
):
    service, engine, _ = setup_service
    body = request()
    body.questions.update(
        StructuredDecisionRequest(
            state="unused",
            questions={
                "urgent": {"type": "noul"},
                "tone": {"type": "score", "criteria": ["calm", "annoyed", "furious"]},
            },
        ).questions
    )
    response = await service.create_decision(body)
    assert isinstance(response, StructuredDecisionResponse)
    assert list(response.answers) == ["team", "urgent", "tone"]
    assert response.answers["team"]["choice"] == "shipping"
    assert response.answers["team"]["probabilities"] == pytest.approx(
        {"billing": 0.2, "shipping": 0.8}
    )
    assert response.answers["urgent"]["noul"] == pytest.approx(0.2)
    tone = response.answers["tone"]
    assert tone["legend"] == {"0": "calm", "1": "annoyed", "2": "furious"}
    assert tone["score"] == pytest.approx(1.2)
    assert all(
        answer["confidence"] == pytest.approx(0.16)
        for answer in response.answers.values()
    )
    assert response.diagnostics["team"].label_mass == pytest.approx(0.2)
    assert all(not diag.argmax_is_label for diag in response.diagnostics.values())
    assert engine.admissions == [3]
    assert response.usage.output_tokens == 3
    assert response.usage.input_tokens == sum(
        len(c[0]["prompt_token_ids"]) for c in engine.calls
    )
    for _, params, _, _ in engine.calls:
        assert isinstance(params, SamplingParams)
        assert params.max_tokens == 1 and params.temperature == 0.0
        assert params.allowed_token_ids is None


@pytest.mark.asyncio
async def test_one_option_confidence_does_not_become_one(setup_service):
    service, _, _ = setup_service
    response = await service.create_decision(
        request(
            questions={
                "only": {"type": "choice", "criteria": {"one": None}},
            }
        )
    )
    assert response.answers["only"]["probabilities"] == {"one": 1.0}
    assert response.answers["only"]["confidence"] == pytest.approx(0.04)


@pytest.mark.asyncio
async def test_metadata_and_unicode_reach_real_v030_renderer_interface(setup_service):
    service, engine, renderer = setup_service
    raw = Request(
        {
            "type": "http",
            "headers": [
                (b"x-request-id", b"from-header"),
                (b"authorization", b"secret"),
                *[(k.encode(), v.encode()) for k, v in TRACE.items()],
            ],
        }
    )
    response = await service.create_decision(
        request(
            instructions="Be accurate.",
            cache_salt="tenant-a",
            priority=7,
            chat_template_kwargs={"custom": 42},
        ),
        raw,
    )
    assert response.id == "decision-from-header"
    prompt, params, rid, metadata = engine.calls[0]
    assert prompt["cache_salt"] == "tenant-a"
    assert rid == "decision-from-header-0"
    assert metadata == {"lora_request": None, "trace_headers": TRACE, "priority": 7}
    messages, chat, tok, extras, skip_mm = renderer.calls[0]
    assert messages[0][0] == {"role": "system", "content": "Be accurate."}
    assert '"message": "Français 日本語"' in messages[0][1]["content"]
    assert chat.chat_template_kwargs["enable_thinking"] is False
    assert chat.chat_template_kwargs["custom"] == 42
    assert chat.chat_template_kwargs["add_generation_prompt"] is True
    assert tok.max_output_tokens == 1 and tok.max_total_tokens == 4096
    assert tok.add_special_tokens is False
    assert extras == {"cache_salt": "tenant-a"} and skip_mm


@pytest.mark.asyncio
async def test_true_label_argmax_is_reported(setup_service):
    service, engine, _ = setup_service
    engine.selected_token = ord("A")
    response = await service.create_decision(request())
    assert response.diagnostics["team"].argmax_is_label is True


@pytest.mark.asyncio
async def test_lora_model_selection_reaches_each_engine_request(setup_service):
    service, engine, _ = setup_service
    adapter = SimpleNamespace(lora_name="qwen-adapter")
    service.models.lora_requests[adapter.lora_name] = adapter
    response = await service.create_decision(request(model=adapter.lora_name))
    assert response.model == adapter.lora_name
    assert engine.calls[0][3]["lora_request"] is adapter


@pytest.mark.asyncio
async def test_render_failure_does_not_leave_partially_submitted_decisions(
    setup_service, monkeypatch
):
    service, engine, renderer = setup_service
    original = renderer.render_chat_async

    async def render(*args, **kwargs):
        if renderer.calls:
            raise VLLMValidationError("second prompt is too long")
        return await original(*args, **kwargs)

    monkeypatch.setattr(renderer, "render_chat_async", render)
    body = request(questions={"one": {"type": "noul"}, "two": {"type": "noul"}})
    with pytest.raises(VLLMValidationError, match="second prompt"):
        await service.create_decision(body)
    assert not engine.calls


@pytest.mark.asyncio
async def test_questions_run_concurrently_and_results_keep_request_order(
    setup_service, monkeypatch
):
    service, engine, _ = setup_service
    original = engine.generate
    ready = asyncio.Event()
    started = []

    async def generate(prompt, params, request_id, **metadata):
        started.append(request_id)
        if len(started) == 2:
            ready.set()
        await asyncio.wait_for(ready.wait(), timeout=1)
        if request_id.endswith("-0"):
            await asyncio.sleep(0)
        async for result in original(prompt, params, request_id, **metadata):
            probability = 0.1 if request_id.endswith("-0") else 0.2
            result.outputs[0].logprobs = [
                {
                    token: Logprob(math.log(probability))
                    for token in params.logprob_token_ids
                }
            ]
            yield result

    monkeypatch.setattr(engine, "generate", generate)
    body = request(questions={"one": {"type": "noul"}, "two": {"type": "noul"}})
    response = await service.create_decision(body)
    assert list(response.answers) == ["one", "two"]
    assert response.answers["one"]["confidence"] == pytest.approx(0.1)
    assert response.answers["two"]["confidence"] == pytest.approx(0.2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"chat_template_kwargs": {"enable_thinking": True}}, "thinking must be off"),
        ({"images": ["data:image/png;base64,AAAA"]}, "does not take image inputs"),
        ({"questions": {}}, "at least one question"),
        (
            {
                "questions": {
                    "q": {
                        "type": "choice",
                        "criteria": {"a": None},
                        "depends_on": ["x"],
                    }
                }
            },
            "unknown field",
        ),
        (
            {"questions": {"q": {"type": "score", "criteria": ["one"]}}},
            "ordered list of levels",
        ),
        (
            {
                "questions": {
                    "q": {
                        "type": "choice",
                        "criteria": dict.fromkeys(map(str, range(27))),
                    }
                }
            },
            "at most 26",
        ),
        ({"questions": {f"q{i}": {"type": "noul"} for i in range(65)}}, "at most 64"),
        ({"model": "missing"}, "does not exist"),
    ],
)
async def test_invalid_requests_fail_before_engine_submission(
    setup_service, overrides, match
):
    service, engine, _ = setup_service
    response = await service.create_decision(request(**overrides))
    assert isinstance(response, ErrorResponse)
    assert match in response.error.message
    assert response.error.code == (404 if overrides.get("model") else 400)
    assert not engine.calls


@pytest.mark.asyncio
async def test_changed_generation_suffix_fails_before_engine_submission(setup_service):
    service, engine, renderer = setup_service
    renderer.bad_tail = True
    response = await service.create_decision(request())
    assert isinstance(response, ErrorResponse)
    assert "end the prompt differently" in response.error.message
    assert not engine.calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,match",
    [
        ("empty_stream", "Failed to generate result"),
        ("empty_outputs", "No output generated"),
        ("engine_error", "Generation error"),
        ("no_logprobs", "No logprobs available"),
        ("missing_label", "not found in logprobs"),
    ],
)
async def test_incomplete_engine_outputs_are_errors(setup_service, failure, match):
    service, engine, _ = setup_service
    engine.failure = failure
    with pytest.raises(ValueError, match=match):
        await service.create_decision(request())


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("mode", ["raw_logprobs", "processed_logprobs"])
def test_official_qwen_architectures_select_next_token(architecture, mode):
    config = SimpleNamespace(architecture=architecture, logprobs_mode=mode)
    assert (
        select_read_strategy(cast("ModelConfig", config), ("generate",))
        is NextTokenStrategy
    )


@pytest.mark.parametrize(
    "architecture,mode,tasks,match",
    [
        (ARCHITECTURES[0], "raw_logits", ("generate",), "not raw_logits"),
        (ARCHITECTURES[0], "processed_logits", ("generate",), "not processed_logits"),
        (ARCHITECTURES[0], "raw_logprobs", ("embed",), "need the generate task"),
        ("Qwen2ForCausalLM", "raw_logprobs", ("generate",), "does not support"),
        ("LlamaForCausalLM", "raw_logprobs", ("generate",), "does not support"),
    ],
)
def test_unsupported_architectures_tasks_and_modes_are_not_enabled(
    architecture, mode, tasks, match
):
    with pytest.raises(ValueError, match=match):
        select_read_strategy(
            cast(
                "ModelConfig",
                SimpleNamespace(architecture=architecture, logprobs_mode=mode),
            ),
            tasks,
        )


def test_backport_registry_is_limited_to_official_qwen_scope():
    assert set(READ_STRATEGIES) == set(ARCHITECTURES)


@pytest.mark.parametrize("salt", ["", "a/b", "a\\b", "a@b", "a\x00b", "x" * 129, 7])
def test_v030_cache_salt_validation_is_preserved(salt):
    with pytest.raises(VLLMValidationError, match="cache_salt"):
        request(cache_salt=salt)


def test_read_prompt_forces_generation_and_reserves_one_output_token():
    read = ReadPromptRequest(chat_template_kwargs={"continue_final_message": True})
    chat = read.build_chat_params(None, "auto").with_defaults({"enable_thinking": True})
    assert chat.chat_template_kwargs["enable_thinking"] is False
    assert chat.chat_template_kwargs["continue_final_message"] is False
    assert chat.chat_template_kwargs["add_generation_prompt"] is True
    tok = read.build_tok_params(cast("ModelConfig", SimpleNamespace(max_model_len=200)))
    assert tok.max_input_tokens == 199


def test_distinct_single_token_labels_are_required():
    tokenizer = cast("TokenizerLike", Tokenizer())
    _, ids = reply_label_ids(tokenizer, [10000, 10, 10])
    assert ids == list(map(ord, LABELS))
    with pytest.raises(ValueError, match="not one distinct token"):
        reply_label_ids(tokenizer, [10000, 10, 10], ("AA", "B"))


def test_router_serves_json_and_preserves_error_status(setup_service):
    service, _, _ = setup_service
    app = FastAPI()
    app.state.serving_structured_decisions = service
    register_structured_decisions_api_router(app)
    with TestClient(app) as client:
        response = client.post("/v1/systemone", json=request().model_dump())
        assert response.status_code == 200
        assert response.json()["answers"]["team"]["choice"] == "shipping"
        response = client.post(
            "/v1/systemone", json=request(model="absent").model_dump()
        )
        assert response.status_code == 404
        assert response.json()["error"]["type"] == "NotFoundError"


def test_service_initialization_enables_supported_qwen_and_disables_unsupported(
    setup_service,
):
    service, engine, _ = setup_service
    state = State(
        {
            "online_renderer": service.strategy.context.online_renderer,
            "openai_serving_models": Models(),
        }
    )
    args = Namespace(chat_template=None, chat_template_content_format="auto")
    init_structured_decisions_state(engine, state, args, None, ("generate",), {})
    assert isinstance(state.serving_structured_decisions, ServingStructuredDecisions)
    engine.model_config.architecture = "LlamaForCausalLM"
    init_structured_decisions_state(engine, state, args, None, ("generate",), {})
    assert state.serving_structured_decisions is None
