"""Build deterministic GSM8K prompt/completion splits for local SFT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_alignment.local_sft import prepare_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(prepare_data(args.seed, args.validation_size), indent=2))


if __name__ == "__main__":
    main()
