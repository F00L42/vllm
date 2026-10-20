# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify copied Qwen semantics against the frozen official v0.32.0 tag."""

import ast
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "5df63a7203cdb9328d824a27b682e77c7b9928be"
BASE = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"
PACKAGE = "vllm/entrypoints/generate/structured_decisions"


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args]).decode("utf-8")


def trees(path):
    original = git("show", f"{SOURCE}:{path}")
    local = (
        (ROOT / path)
        .read_text(encoding="utf-8")
        .replace("vllm.entrypoints.chat_utils", "vllm.renderers.chat_utils")
    )
    return ast.parse(local), ast.parse(original)


def definition(tree, name):
    return next(node for node in tree.body if getattr(node, "name", None) == name)


def check(actual, expected, label, checked):
    if isinstance(actual, ast.Module) and isinstance(expected, ast.Module):
        # Ruff reorders the relocated import; declaration order stays unchanged.
        for tree in (actual, expected):
            imports = [
                n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
            ]
            other = [
                n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))
            ]
            tree.body = sorted(imports, key=ast.dump) + other
    assert ast.dump(actual) == ast.dump(expected), f"Upstream semantics differ: {label}"
    checked.append(label)


def main():
    assert git("rev-parse", "v0.32.0").strip() == SOURCE
    assert git("rev-parse", "v0.30.0").strip() == BASE
    subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", BASE, "HEAD"],
        check=True,
    )
    checked = []
    for path in (
        "vllm/entrypoints/generate/label_reads.py",
        f"{PACKAGE}/question_types.py",
        f"{PACKAGE}/protocol.py",
        f"{PACKAGE}/api_router.py",
    ):
        actual, expected = trees(path)
        check(actual, expected, path, checked)

    actual, expected = trees(f"{PACKAGE}/strategies.py")
    for name in (
        "ReadContext",
        "DecisionLimits",
        "QuestionRead",
        "state_text",
        "ReadStrategy",
        "prompt_tail",
        "reply_label_ids",
        "type_label_ids",
        "cap_pixels",
        "NextTokenStrategy",
        "select_read_strategy",
    ):
        check(definition(actual, name), definition(expected, name), name, checked)
    for name in ("READ_STRATEGIES", "LOGPROBS_MODES"):
        # Inspect literal registry/config without importing the inference engine.
        def assignment(tree, name=name):
            for node in tree.body:
                if isinstance(node, ast.AnnAssign) and node.target.id == name:
                    return node.value
                if isinstance(node, ast.Assign) and node.targets[0].id == name:
                    return node.value
            raise AssertionError(name)

        actual_value, expected_value = assignment(actual), assignment(expected)
        if name == "READ_STRATEGIES":
            assert isinstance(expected_value, ast.Dict)
            pairs = [
                (key, value)
                for key, value in zip(expected_value.keys, expected_value.values)
                if key.value.startswith("Qwen")
            ]
            expected_value = ast.Dict(
                keys=[key for key, _ in pairs], values=[value for _, value in pairs]
            )
        check(actual_value, expected_value, name, checked)

    actual, expected = trees(f"{PACKAGE}/serving.py")
    check(
        definition(actual, "parse_questions"),
        definition(expected, "parse_questions"),
        "parse_questions",
        checked,
    )
    for name in ("__init__", "_media", "create_decision"):
        check(
            definition(definition(actual, "ServingStructuredDecisions"), name),
            definition(definition(expected, "ServingStructuredDecisions"), name),
            f"ServingStructuredDecisions.{name}",
            checked,
        )
    trace = ast.parse(git("show", f"{SOURCE}:vllm/entrypoints/serve/engine/serving.py"))
    check(
        definition(
            definition(actual, "ServingStructuredDecisions"), "_get_trace_headers"
        ),
        definition(definition(trace, "BaseServing"), "_get_trace_headers"),
        "BaseServing._get_trace_headers (local compatibility method)",
        checked,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "base": BASE,
                "source": SOURCE,
                "checked": checked,
                "boundary": "Static source parity; no engine or NPU execution.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
