"""Standalone runner for FunctionJob. STDLIB ONLY: it is copied into the task workspace
and executed by the worker's workload interpreter, which need not have gpulayer installed.

usage: python function_runner.py <call.json> <result.json>
"""

import importlib
import json
import sys
import traceback


def main() -> int:
    call_path, result_path = sys.argv[1], sys.argv[2]
    with open(call_path) as f:
        call = json.load(f)
    module_name, func_name = call["entrypoint"].split(":")
    try:
        func = getattr(importlib.import_module(module_name), func_name)
        value = func(*call["args"], **call["kwargs"])
        payload = json.dumps({"ok": True, "value": value})
    except TypeError as e:
        if "JSON serializable" in str(e):
            payload = json.dumps(
                {"ok": False, "error": f"return value is not JSON-serializable: {e}"}
            )
        else:
            payload = json.dumps({"ok": False, "error": traceback.format_exc()})
    except BaseException:  # noqa: BLE001 - report everything, including SystemExit
        payload = json.dumps({"ok": False, "error": traceback.format_exc()})
    with open(result_path, "w") as f:
        f.write(payload)
    return 0 if json.loads(payload)["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
