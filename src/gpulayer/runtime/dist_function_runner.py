"""Standalone per-rank runner for a function-based DistributedJob. STDLIB ONLY: copied
into the task workspace and launched by torchrun once per local rank.

usage: torchrun ... dist_function_runner.py <call.json> <result.json>

Every rank calls the function; only global rank 0 writes the result file. An exception
on any rank exits non-zero, which makes torchrun (and so the whole job) fail.
"""

import importlib
import json
import os
import sys


def main() -> int:
    call_path, result_path = sys.argv[1], sys.argv[2]
    with open(call_path) as f:
        call = json.load(f)
    module_name, func_name = call["entrypoint"].split(":")
    func = getattr(importlib.import_module(module_name), func_name)
    value = func(*call["args"], **call["kwargs"])
    if os.environ.get("RANK", "0") == "0":
        try:
            payload = json.dumps({"ok": True, "value": value})
        except TypeError as e:
            payload = json.dumps(
                {"ok": False, "error": f"return value is not JSON-serializable: {e}"}
            )
        with open(result_path, "w") as f:
            f.write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
