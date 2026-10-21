"""Make the repo root importable so `import vcb...` works from tests/, and keep
tests hermetic: the developer's own .env / VCB_* environment must not leak in."""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _key in [k for k in os.environ if k.startswith("VCB_")]:
    del os.environ[_key]
os.environ["VCB_ENV_FILE"] = str(_ROOT / "tests" / "_no_such.env")
