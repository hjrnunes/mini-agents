"""Write (or with --check, verify) src/mini_agents/declarations/."""

from __future__ import annotations

import argparse
import logging
import sys

from mini_agents.kernel.declarations import rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if a file is stale.")
    args = parser.parse_args(argv)
    # Each tools/list request would log a line.
    logging.disable(logging.INFO)
    stale = []
    for path, text in rendered().items():
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for path in stale:
        print(
            f"gen_declarations.py: {path} is stale; run scripts/gen_declarations.py",
            file=sys.stderr,
        )
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
