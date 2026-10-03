import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if os.environ.get("SWM_PATH"):  # a stable-worldmodel checkout that is not pip-installed
    sys.path.insert(0, os.environ["SWM_PATH"])
