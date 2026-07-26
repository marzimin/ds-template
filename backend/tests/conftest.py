from pathlib import Path

# Ensure the project-root .env exists before importing project code that reads
# it at import time. Anchored to this file rather than the working directory so
# the suite behaves the same from the repo root or from backend/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
(PROJECT_ROOT / ".env").touch(exist_ok=True)
