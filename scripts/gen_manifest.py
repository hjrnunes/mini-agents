"""Write (or with --check, verify) src/mini_agents/tool-manifest.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mini_agents.kernel.manifest import MANIFEST_PATH, ManifestError, render_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the file is stale.")
    parser.add_argument("--path", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args(argv)
    try:
        text = render_manifest()
    except ManifestError as exc:
        print(f"gen_manifest.py: {exc}", file=sys.stderr)
        return 1
    if not args.check:
        args.path.write_text(text)
        return 0
    if not args.path.exists() or args.path.read_text() != text:
        print(
            f"gen_manifest.py: {args.path} is stale; run scripts/gen_manifest.py",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
