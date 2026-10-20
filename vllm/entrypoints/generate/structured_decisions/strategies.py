# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read strategies: how a model is asked for label probabilities.

Only models in READ_STRATEGIES are currently supported.
"""

import asyncio
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar

from PIL import Image

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.generate.label_reads import next_token_label_reads
from vllm.inputs import EngineInput
from vllm.lora.request import LoRARequest
from vllm.multimodal.media import (
    MEDIA_CONNECTOR_REGISTRY,
    MediaConnector,
)
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.tokenizers import TokenizerLike

from .protocol import ReadPromptRequest
from .question_types import (
    LABELS,
    QUESTION_TYPES,
    Question,
    StructuredDecisionError,
    label_softmax,
)


@dataclass
class ReadContext:
    engine_client: EngineClient
    online_renderer: OnlineRenderer
    chat_template: str | None
    chat_template_content_format: ChatTemplateContentFormatOption
    default_chat_template_kwargs: dict[str, Any]


@dataclass(frozen=True)
class DecisionLimits:
    max_questions: int
    max_options: int


@dataclass
class QuestionRead:
    probs: list[float]
    label_mass: float
    argmax_is_label: bool
    input_tokens: int
    output_tokens: int


def state_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


class ReadStrategy(ABC):
    #: The engine task reads run as.
    task: ClassVar[SupportedTask] = "generate"
    #: The modalities a request may attach media of.
    modalities: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, context: ReadContext):
        self.context = context

    @cached_property
    def _connector(self) -> MediaConnector:
        model_config = self.context.engine_client.model_config
        return MEDIA_CONNECTOR_REGISTRY.load(
            envs.VLLM_MEDIA_CONNECTOR,
            media_io_kwargs=model_config.get_multimodal_config().media_io_kwargs,
            allowed_local_media_path=model_config.allowed_local_media_path,
            allowed_media_domains=model_config.allowed_media_domains,
        )

    @abstractmethod
    def limits(self) -> DecisionLimits: ...

    def render_state(self, state: Any) -> str:
        """The request's state as the model reads it."""
        return state_text(state)

    def num_requests(self, questions: list[Question]) -> int:
        """The engine requests a decision takes."""
        return len(questions)

    @abstractmethod
    async def read(
        self,
        questions: list[Question],
        instructions: str | None,
        state: str,
        *,
        request_id: str,
        chat_template_kwargs: dict[str, Any] | None,
        lora_request: LoRARequest | None,
        priority: int,
        cache_salt: str | None = None,
        trace_headers: Mapping[str, str] | None = None,
        media: Mapping[str, Sequence[str]] | None = None,
    ) -> list[QuestionRead]:
        """One read per question, in order."""


def prompt_tail(tokenizer: TokenizerLike, prompt_ids: Sequence[int]) -> list[int]:
    """The prompt's ids after its last added token."""
    added = set(tokenizer.get_added_vocab().values())
    start = max((i + 1 for i, t in enumerate(prompt_ids) if t in added), default=0)
    return list(prompt_ids[start:])


def reply_label_ids(
    tokenizer: TokenizerLike,
    prompt_ids: Sequence[int],
    labels: Sequence[str] = LABELS,
) -> tuple[list[int], list[int]]:
    """The prompt's ids after its last added token, and the token each of
    ``labels`` adds after them as the first token of the reply. Raises
    ValueError if a label is not one distinct token there."""
    tail = prompt_tail(tokenizer, prompt_ids)
    text = tokenizer.decode(tail)
    ids: list[int] = []
    for label in labels:
        extended = tokenizer.encode(text + label, add_special_tokens=False)
        if extended[:-1] != tail or extended[-1] in ids:
            raise ValueError(
                f"label {label!r} is not one distinct token after this model's "
                "chat prompt"
            )
        ids.append(extended[-1])
    return tail, ids


def type_label_ids(
    tokenizer: TokenizerLike, prompt_ids: Sequence[int]
) -> dict[str, int]:
    """The reply's first token for each label of every registered question
    type. Raises ValueError as reply_label_ids does."""
    label_ids: dict[str, int] = {}
    for qtype in QUESTION_TYPES.values():
        _, ids = reply_label_ids(tokenizer, prompt_ids, qtype.label_set)
        label_ids.update(zip(qtype.label_set, ids))
    return label_ids


def cap_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    """Shrink ``image`` to at most ``max_pixels`` pixels (bicubic)."""
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (w * h))
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return image.resize(size, Image.Resampling.BICUBIC)


class NextTokenStrategy(ReadStrategy):
    """Autoregressive models. Each question is one request: the state, then the
    question with its labeled options, and the logprobs of the label tokens as
    the reply's first token. The requests share the state. With prefix caching,
    it is prefilled once."""

    #: Images are shrunk to at most this many pixels before the processor.
    max_image_pixels: ClassVar[int | None] = None

    def __init__(self, context: ReadContext):
        super().__init__(context)
        # Every prompt ends with the same generation prompt, so a label's token
        # is the same for every question.
        prompt = self._generation_prompt()
        self.tail = prompt_tail(self._tokenizer(), prompt)
        self.label_ids = type_label_ids(self._tokenizer(), prompt)

    def _tokenizer(self) -> TokenizerLike:
        return self.context.online_renderer.renderer.get_tokenizer()

    def _generation_prompt(self) -> list[int]:
        """A chat prompt's ids up to the generation prompt, with thinking off."""
        context, tokenizer = self.context, self._tokenizer()
        probe = tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}],
            **{
                **context.default_chat_template_kwargs,
                "chat_template": context.chat_template,
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": False,
                "enable_thinking": False,
            },
        )
        if isinstance(probe, str):
            probe = tokenizer.encode(probe, add_special_tokens=False)
        return list(probe)

    def _sampling_params(
        self, label_ids: list[int], prompt_ids: list[int]
    ) -> SamplingParams:
        return SamplingParams(
            max_tokens=1, temperature=0.0, logprob_token_ids=label_ids
        )

    def _read_input(
        self, engine_input: EngineInput, prompt_ids: list[int]
    ) -> EngineInput:
        """The engine input for a read of the rendered prompt."""
        return engine_input

    def _label_groups(self, question: Question) -> list[list[int]]:
        """The tokens that count for each label. A label scores its best one."""
        return [[self.label_ids[label]] for label in question.labels]

    def _content(self, state: str, question: Question) -> str:
        """The user turn: the state, then the question."""
        return f"{state}\n\n{question.type.prompt(question)}"

    async def _image_parts(self, urls: Sequence[str]) -> list[dict[str, Any]]:
        """The images as chat content parts, fetched once so that every
        question reads the same pictures."""
        if not urls:
            return []
        try:
            images = await asyncio.gather(*map(self._connector.fetch_image_async, urls))
        except ValueError as e:
            raise StructuredDecisionError(f"media: {e}") from e
        if self.max_image_pixels is not None:
            images = [cap_pixels(image, self.max_image_pixels) for image in images]
        return [{"type": "image_pil", "image_pil": image} for image in images]

    def limits(self) -> DecisionLimits:
        return DecisionLimits(max_questions=64, max_options=len(LABELS))

    def _read_request(
        self, chat_template_kwargs: dict[str, Any] | None, cache_salt: str | None = None
    ) -> ReadPromptRequest:
        if (chat_template_kwargs or {}).get("enable_thinking"):
            raise StructuredDecisionError(
                "a read is the reply's first token, so thinking must be off"
            )
        return ReadPromptRequest(
            chat_template_kwargs=chat_template_kwargs, cache_salt=cache_salt
        )

    async def _render(
        self, read_request: ReadPromptRequest, messages: list[dict[str, Any]]
    ) -> tuple[EngineInput, list[int]]:
        ctx = self.context
        _, (engine_input,) = await ctx.online_renderer.preprocess_chat(
            read_request,
            messages,
            default_template=ctx.chat_template,
            default_template_content_format=ctx.chat_template_content_format,
            default_template_kwargs=ctx.default_chat_template_kwargs,
            # A read that fails before it reaches the engine must not leave the
            # sender cache believing the engine has the images.
            skip_mm_cache=True,
        )
        prompt_ids = extract_prompt_components(
            ctx.engine_client.model_config, engine_input
        ).token_ids
        return engine_input, list(prompt_ids or [])

    async def read(
        self,
        questions: list[Question],
        instructions: str | None,
        state: str,
        *,
        request_id: str,
        chat_template_kwargs: dict[str, Any] | None,
        lora_request: LoRARequest | None,
        priority: int,
        cache_salt: str | None = None,
        trace_headers: Mapping[str, str] | None = None,
        media: Mapping[str, Sequence[str]] | None = None,
    ) -> list[QuestionRead]:
        ctx = self.context
        read_request = self._read_request(chat_template_kwargs, cache_salt)
        image_parts = await self._image_parts((media or {}).get("image", ()))

        slots, engine_inputs, params = [], [], []
        for q in questions:
            content: Any = self._content(state, q)
            if image_parts:
                content = [*image_parts, {"type": "text", "text": content}]
            messages = [{"role": "user", "content": content}]
            if instructions:
                messages.insert(0, {"role": "system", "content": instructions})
            engine_input, prompt_ids = await self._render(read_request, messages)
            if prompt_ids[len(prompt_ids) - len(self.tail) :] != self.tail:
                raise StructuredDecisionError(
                    "these chat options end the prompt differently, so the "
                    "labels' tokens are unknown"
                )
            slots.append(self._label_groups(q))
            engine_inputs.append(self._read_input(engine_input, prompt_ids))
            params.append(
                self._sampling_params([i for ids in slots[-1] for i in ids], prompt_ids)
            )

        label_reads = await next_token_label_reads(
            ctx.engine_client,
            engine_inputs,
            params,
            request_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=priority,
        )

        reads = []
        for groups, label_read in zip(slots, label_reads):
            output = label_read.result.outputs[0]
            logprobs = iter(label_read.logprobs)
            reads.append(
                QuestionRead(
                    probs=label_softmax(
                        [max(next(logprobs) for _ in g) for g in groups]
                    ),
                    label_mass=sum(math.exp(lp) for lp in label_read.logprobs),
                    argmax_is_label=bool(output.token_ids)
                    and any(output.token_ids[0] in ids for ids in groups),
                    input_tokens=len(label_read.result.prompt_token_ids or ()),
                    output_tokens=len(output.token_ids),
                )
            )
        return reads


READ_STRATEGIES: dict[str, type[ReadStrategy]] = {
    "Qwen3ForCausalLM": NextTokenStrategy,
    "Qwen3_5ForConditionalGeneration": NextTokenStrategy,
    "Qwen3_5MoeForConditionalGeneration": NextTokenStrategy,
}


# label_mass sums the labels' full-vocabulary probabilities. The logprobs modes
# return those probabilities. The logits modes return raw logits instead.
LOGPROBS_MODES = frozenset({"raw_logprobs", "processed_logprobs"})


def select_read_strategy(
    model_config: ModelConfig, supported_tasks: Sequence[SupportedTask]
) -> type[ReadStrategy]:
    """Raise ValueError when the model cannot serve structured decisions."""
    strategy = READ_STRATEGIES.get(model_config.architecture)
    if strategy is None:
        raise ValueError(
            "The structured decisions API does not support "
            f"{model_config.architecture}. Supported architectures: "
            f"{sorted(READ_STRATEGIES)}"
        )
    if strategy.task not in supported_tasks:
        raise ValueError(
            f"Structured decisions on {model_config.architecture} need the "
            f"{strategy.task} task"
        )
    if strategy.task == "generate" and model_config.logprobs_mode not in LOGPROBS_MODES:
        raise ValueError(
            "Structured decisions need --logprobs-mode raw_logprobs or "
            f"processed_logprobs, not {model_config.logprobs_mode}"
        )
    return strategy
