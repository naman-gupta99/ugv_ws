#!/usr/bin/env python3

import argparse
import sys

from langchain_core.messages import HumanMessage

from .agent.models import Models


def _has_text_response(response) -> bool:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return bool(content)
    return content is not None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify configured inspection LLMs return a response."
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="Specific model names to test. Defaults to every model in Models.",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: OK",
        help="Prompt used for the smoke test.",
    )
    args = parser.parse_args(argv)

    registry = Models()
    model_names = args.models or registry.available_model_names()
    llm_model_names = set(registry.llm_model_names())

    failures = []
    for model_name in model_names:
        if model_name not in llm_model_names:
            print(f"[model-smoke-test] SKIP {model_name}: classical baseline", flush=True)
            continue
        print(f"[model-smoke-test] Testing {model_name}...", flush=True)
        try:
            response = registry.get_model(model_name).invoke(
                [HumanMessage(content=args.prompt)]
            )
            if not _has_text_response(response):
                raise RuntimeError("empty response")
        except Exception as exc:
            failures.append((model_name, str(exc)))
            print(f"[model-smoke-test] FAIL {model_name}: {exc}", flush=True)
        else:
            print(f"[model-smoke-test] OK {model_name}", flush=True)

    if failures:
        print("\n[model-smoke-test] One or more models failed:", file=sys.stderr)
        for model_name, reason in failures:
            print(f"  - {model_name}: {reason}", file=sys.stderr)
        return 1

    print("\n[model-smoke-test] All requested models returned responses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
