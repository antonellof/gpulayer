"""Write env.sh for clients and workers from a `gpulayer up` home. Used by gcp.sh on the server.

python write_env.py GPULAYER_HOME URL FINGERPRINT > env.sh
"""

import json
import sys
from pathlib import Path

home, url, fingerprint = sys.argv[1:4]
tokens = json.loads(Path(home, "up-tokens.json").read_text())


def pick(*words: str) -> str:
    for w in words:
        for k, v in tokens.items():
            if w in k:
                return v
    raise SystemExit(f"no {words} token in up-tokens.json")


print(f"export GPULAYER_URL={url}")
print(f"export GPULAYER_FINGERPRINT={fingerprint}")
print(f"export GPULAYER_TOKEN={pick('client', 'admin')}")
print(f"export GPULAYER_JOIN_TOKEN={pick('join')}")
