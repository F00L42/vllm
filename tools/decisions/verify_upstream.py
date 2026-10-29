# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify original semantics, the pinned warm source, and declared extensions."""

import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "5df63a7203cdb9328d824a27b682e77c7b9928be"
BASE = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"
BACKPORT_SOURCE = "ea2e3575d07bf49fdefa1e74ceb349896ff74f8d"
PERF_SOURCE = "ee22f682ab0d2139ae69e3cdbb762a565a166813"
PACKAGE = "vllm/entrypoints/generate/structured_decisions"


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args]).decode("utf-8")


def definition(tree, name):
    return next(node for node in tree.body if getattr(node, "name", None) == name)


def symbol(tree, name):
    for part in name.split("."):
        try:
            tree = definition(tree, part)
        except StopIteration:
            tree = next(
                node
                for node in tree.body
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                and assignment_name(node) == part
            )
    return tree


def assignment_name(node):
    if isinstance(node, ast.AnnAssign):
        return getattr(node.target, "id", None)
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        return getattr(node.targets[0], "id", None)
    return None


def same(actual, expected, label, checked):
    assert ast.dump(actual) == ast.dump(expected), f"Unregistered difference: {label}"
    checked.append(label)


def tree(path, ref=None):
    text = git("show", f"{ref}:{path}") if ref else (ROOT / path).read_text("utf-8")
    return ast.parse(
        text.replace("vllm.entrypoints.chat_utils", "vllm.renderers.chat_utils")
    )


def strip_imports(module):
    module.body = [
        node
        for node in module.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def main():
    checked = []
    assert git("rev-parse", "v0.30.0").strip() == BASE
    assert git("rev-parse", "v0.32.0").strip() == SOURCE
    git("merge-base", "--is-ancestor", BASE, "HEAD")
    manifest = json.loads((ROOT / "tools/decisions/extensions.json").read_text("utf-8"))
    assert manifest["source"] == SOURCE and manifest["perf_source"] == PERF_SOURCE
    for path, changes in manifest["symbols"].items():
        current = tree(path)
        for name, contract in changes.items():
            actual = hashlib.sha256(
                ast.dump(symbol(current, name)).encode()
            ).hexdigest()
            assert actual == contract["ast_sha256"], (
                f"Extension contract drift: {path}:{name}"
            )
            checked.append(f"declared extension: {path}:{name}")

    # Every unmodified declaration still matches the original release. Modified
    # symbols are individually frozen above and covered by behavioral tests.
    for path in (
        f"{PACKAGE}/protocol.py",
        f"{PACKAGE}/question_types.py",
        f"{PACKAGE}/api_router.py",
        "vllm/entrypoints/generate/label_reads.py",
        f"{PACKAGE}/strategies.py",
    ):
        actual, expected = tree(path), tree(path, BACKPORT_SOURCE)
        imports = [
            node
            for node in actual.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        digest = hashlib.sha256(
            "\n".join(sorted(map(ast.dump, imports))).encode()
        ).hexdigest()
        assert digest == manifest["imports"][path], f"Import contract drift: {path}"
        changes = manifest["symbols"].get(path, {})
        for name in changes:
            parts = name.split(".")
            actual_parent, expected_parent = actual, expected
            for part in parts[:-1]:
                actual_parent = definition(actual_parent, part)
                expected_parent = definition(expected_parent, part)
            changed = symbol(actual_parent, parts[-1])
            index = actual_parent.body.index(changed)
            actual_parent.body.pop(index)
            try:
                original = symbol(expected_parent, parts[-1])
            except StopIteration:
                continue
            actual_parent.body.insert(index, copy.deepcopy(original))
        strip_imports(actual)
        strip_imports(expected)
        same(actual, expected, path, checked)

    actual, expected = (
        tree(f"{PACKAGE}/strategies.py"),
        tree(f"{PACKAGE}/strategies.py", SOURCE),
    )
    for name in (
        "ReadContext",
        "DecisionLimits",
        "QuestionRead",
        "state_text",
        "ReadStrategy",
        "prompt_tail",
        "cap_pixels",
        "select_read_strategy",
    ):
        same(definition(actual, name), definition(expected, name), name, checked)
    current_class = copy.deepcopy(definition(actual, "NextTokenStrategy"))
    original_class = definition(expected, "NextTokenStrategy")
    for name in manifest["symbols"][f"{PACKAGE}/strategies.py"]:
        if not name.startswith("NextTokenStrategy."):
            continue
        method = name.split(".")[1]
        changed = definition(current_class, method)
        index = current_class.body.index(changed)
        current_class.body.pop(index)
        try:
            original = definition(original_class, method)
        except StopIteration:
            continue
        current_class.body.insert(index, copy.deepcopy(original))
    same(current_class, original_class, "unchanged NextTokenStrategy methods", checked)
    # The registry is intentionally the same three Qwen entries as the baseline.
    registry = symbol(actual, "READ_STRATEGIES").value
    reference = symbol(expected, "READ_STRATEGIES").value
    reference.keys, reference.values = zip(
        *(
            (key, value)
            for key, value in zip(reference.keys, reference.values)
            if key.value.startswith("Qwen")
        )
    )
    reference.keys, reference.values = list(reference.keys), list(reference.values)
    same(registry, reference, "Qwen registry", checked)
    same(
        symbol(actual, "LOGPROBS_MODES"),
        symbol(expected, "LOGPROBS_MODES"),
        "logprobs modes",
        checked,
    )

    perf = tree("vllm/entrypoints/openai/decisions/strategies.py", PERF_SOURCE)
    prefix = copy.deepcopy(definition(actual, "common_prefix_len"))
    # The local helper handles [] explicitly; the nonempty algorithm is exact.
    empty = prefix.body.pop(1)
    assert ast.dump(empty) == ast.dump(
        ast.parse("if not prompts:\n    return 0").body[0]
    )
    same(
        prefix,
        definition(perf, "common_prefix_len"),
        "PERF_SOURCE: common_prefix_len",
        checked,
    )
    same(
        symbol(actual, "MIN_WARM_PREFIX_TOKENS"),
        symbol(perf, "MIN_WARM_PREFIX_TOKENS"),
        "PERF_SOURCE: threshold",
        checked,
    )

    def producer_call(node):
        return next(
            item
            for item in ast.walk(node)
            if isinstance(item, ast.Assign)
            and len(item.targets) == 1
            and getattr(item.targets[0], "id", None) == "generator"
        )

    same(
        producer_call(
            definition(definition(actual, "NextTokenStrategy"), "_warm_shared_prefix")
        ),
        producer_call(
            definition(definition(perf, "NextTokenStrategy"), "_warm_shared_prefix")
        ),
        "PERF_SOURCE: producer input/params/metadata",
        checked,
    )
    read = definition(definition(actual, "NextTokenStrategy"), "read")
    assert any(
        isinstance(call, ast.Attribute) and call.attr == "_warm_shared_prefix"
        for call in ast.walk(read)
    )
    assert "enforce_eager" not in ast.unparse(
        read
    ) and "async_scheduling" not in ast.unparse(read)

    actual, expected = (
        tree(f"{PACKAGE}/serving.py"),
        tree(f"{PACKAGE}/serving.py", SOURCE),
    )
    same(
        definition(actual, "parse_questions"),
        definition(expected, "parse_questions"),
        "parse_questions",
        checked,
    )
    for method in ("__init__", "_media", "create_decision"):
        same(
            definition(definition(actual, "ServingStructuredDecisions"), method),
            definition(definition(expected, "ServingStructuredDecisions"), method),
            f"service: {method}",
            checked,
        )
    trace = tree("vllm/entrypoints/serve/engine/serving.py", SOURCE)
    same(
        definition(
            definition(actual, "ServingStructuredDecisions"), "_get_trace_headers"
        ),
        definition(definition(trace, "BaseServing"), "_get_trace_headers"),
        "trace compatibility method",
        checked,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "source": SOURCE,
                "perf_source": PERF_SOURCE,
                "checked": checked,
                "boundary": "Source contracts only; engine/NPU validation is separate.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
