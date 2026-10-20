# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real tokenizer contracts from v0.32.0 test_strategies.py.

Requires the Qwen tokenizer files and a complete vLLM Python environment.
No model weights or accelerator are needed for these tokenizer checks.
"""

import pytest

from vllm.entrypoints.generate.structured_decisions.question_types import LABELS
from vllm.entrypoints.generate.structured_decisions.strategies import reply_label_ids


@pytest.fixture(scope="module")
def qwen():
    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer("Qwen/Qwen3-0.6B")
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "state"}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=False,
    )
    return tokenizer, prompt_ids


def test_labels_start_the_reply(qwen):
    tokenizer, prompt_ids = qwen
    tail, ids = reply_label_ids(tokenizer, prompt_ids)
    assert tokenizer.decode(tail) == "\n\n"
    assert [tokenizer.decode([i]) for i in ids] == list(LABELS)
    _, ids = reply_label_ids(tokenizer, prompt_ids, ("yes", "no", *"0123456789"))
    assert len(set(ids)) == 12
    with pytest.raises(ValueError, match="not one distinct token"):
        reply_label_ids(tokenizer, tokenizer.encode("team:"))
