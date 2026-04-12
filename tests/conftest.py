from pathlib import Path

# Ensure .env exists before importing project code that reads it at import time
Path(".env").touch(exist_ok=True)
