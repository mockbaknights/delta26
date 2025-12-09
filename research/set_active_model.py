from __future__ import annotations

import argparse
import json
from pathlib import Path

REGISTRY_PATH = Path("research/models/model_registry.json")


def set_active(version: str) -> None:
    if not REGISTRY_PATH.exists():
        raise SystemExit("Registry not found at research/models/model_registry.json")
    registry = json.loads(REGISTRY_PATH.read_text())
    versions = {v["version"]: v for v in registry.get("versions", [])}
    if version not in versions:
        raise SystemExit(f"Version {version} not found in registry.")
    registry["active_version"] = version
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))
    print(f"Active model set to {version}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="Version to activate, e.g., v2")
    args = parser.parse_args()
    set_active(args.version)


if __name__ == "__main__":
    main()

