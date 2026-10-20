"""Write (or with --check, verify) src/mini_agents/midojo-base.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mini_agents.kernel.midojo_base import BASE_PATH, render_base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Fail if the file is stale."
    )
    parser.add_argument("--path", type=Path, default=BASE_PATH)
    args = parser.parse_args(argv)
    try:
        text = render_base()
    except ValueError as exc:
        print(f"gen_midojo_base.py: {exc}", file=sys.stderr)
        return 1
    if not args.check:
        args.path.write_text(text)
        return 0
    if not args.path.exists() or args.path.read_text() != text:
        print(
            f"gen_midojo_base.py: {args.path} is stale; run scripts/gen_midojo_base.py",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
