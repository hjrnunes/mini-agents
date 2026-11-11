"""Patch the installed OGX where the mini-agents gateway meets two defects.

streaming.py: OGX's streaming tool-call merge for vLLM Hermes-style deltas.
vLLM (Hermes tool parser) streams tool calls in shapes OGX's chunk-merge
does not tolerate:

- The function name may arrive in a later delta than the first chunk, or be
  absent entirely for zero-argument calls. OGX constructs the accumulated
  tool call from the first chunk and rejects function.name=None (pydantic
  string_type), and never backfills the name on later chunks.
- The zero-argument shape ({} arguments) arrives as a single delta with no
  id and no name. OGX cannot recover the tool name from such a stream; the
  hallucinated-tool fallback exists for exactly this case, but it crashes
  first on tool_call_id=None.

tool_executor.py: the error text of an MCP tool result. OGX's MCP runtime
sets error_code=1 for an isError result and never sets error_message, so the
call item's `error` reads "Error (code 1): None" and the server's text, which
sits in the result content, never reaches it.

Each replacement carries a marker substring that appears only in the patched
text, so the script is idempotent and safe to re-run after partial upgrades.

A replacement whose pre-patch text is gone, or matches more than once, is an
error: the script prints "no longer matches the pre-patch text" (or the match
count) to stderr, writes no file and exits 1. `--check` changes nothing and
exits 1 unless every marker is present; the sandbox image build and
smoke_sandbox_image.sh use it.

Removal condition: pyproject.toml pins ogx to the release this script fits
(1.3.1). Remove the streaming replacements when the locked OGX release handles
both delta shapes above unpatched, and the tool_executor replacement when it
keeps an MCP isError result's text in `error`. Remove the script when no
replacement is left. A rewritten block fails with the same message, so confirm
with an unpatched Hermes stream or an unpatched isError result before you
remove a replacement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# (old, new, marker) — marker is a substring unique to the patched text.
STREAMING_REPLACEMENTS: list[tuple[str, str, str]] = [
    # 1. First-chunk construction: tolerate name=None and id=None.
    (
        """                            # arguments may be None in the first streaming delta (name/index arrive before arguments)
                            # Initialize to "" so subsequent argument chunks accumulate correctly.
                            # The final "{}" fallback is applied at the end of streaming.
                            if tool_call_dict.get("function") and tool_call_dict["function"].get("arguments") is None:
                                tool_call_dict["function"]["arguments"] = ""
                            response_tool_call = OpenAIChatCompletionToolCall(**tool_call_dict)
""",
        """                            # arguments may be None in the first streaming delta (name/index arrive before arguments)
                            # Initialize to "" so subsequent argument chunks accumulate correctly.
                            # The final "{}" fallback is applied at the end of streaming.
                            # vLLM's zero-arg tool-call shape omits id/name entirely; default
                            # both so the first-chunk construction succeeds and they can be
                            # backfilled on a later chunk.
                            if tool_call_dict.get("id") is None:
                                tool_call_dict["id"] = ""
                            # vLLM Hermes-style streaming may also send the function name in a
                            # later delta than the first chunk; tolerate name=None here and
                            # backfill it on a subsequent chunk.
                            if tool_call_dict.get("function"):
                                if tool_call_dict["function"].get("arguments") is None:
                                    tool_call_dict["function"]["arguments"] = ""
                                if tool_call_dict["function"].get("name") is None:
                                    tool_call_dict["function"]["name"] = ""
                            response_tool_call = OpenAIChatCompletionToolCall(**tool_call_dict)
""",
        'omits id/name entirely; default',
    ),
    # 2. Accumulate path: backfill the function name on later chunks.
    (
        """                            # Accumulate arguments for final response (only for subsequent chunks)
                            if not is_new_tool_call and response_tool_call is not None:
                                # Both should have functions since we're inside the tool_call.function check above
                                assert response_tool_call.function is not None
                                assert tool_call.function is not None
                                response_tool_call.function.arguments = (
                                    response_tool_call.function.arguments or ""
                                ) + tool_call.function.arguments
""",
        """                            # Accumulate arguments for final response (only for subsequent chunks)
                            # The function name may arrive in a later delta than the first
                            # chunk; backfill it when the accumulated entry is still unnamed.
                            if (
                                not is_new_tool_call
                                and response_tool_call is not None
                                and response_tool_call.function is not None
                                and tool_call.function is not None
                                and tool_call.function.name
                                and not response_tool_call.function.name
                            ):
                                response_tool_call.function.name = tool_call.function.name
                            if not is_new_tool_call and response_tool_call is not None:
                                # Both should have functions since we're inside the tool_call.function check above
                                assert response_tool_call.function is not None
                                assert tool_call.function is not None
                                response_tool_call.function.arguments = (
                                    response_tool_call.function.arguments or ""
                                ) + tool_call.function.arguments
""",
        'backfill it when the accumulated entry is still unnamed',
    ),
    # 3. item.added stream event: tolerate name=None.
    (
        """                                function_call_item = OpenAIResponseOutputMessageFunctionToolCall(
                                    arguments="",  # Will be filled incrementally via delta events
                                    call_id=tool_call.id or "",
                                    name=tool_call.function.name if tool_call.function else "",
""",
        """                                function_call_item = OpenAIResponseOutputMessageFunctionToolCall(
                                    arguments="",  # Will be filled incrementally via delta events
                                    call_id=tool_call.id or "",
                                    # name may arrive in a later delta than the first chunk (vLLM
                                    # Hermes-style streaming); tolerate name=None here.
                                    name=tool_call.function.name or "" if tool_call.function else "",
""",
        'tolerate name=None here.',
    ),
    # 4. Tool execution: backfill the accumulated entry's id when the provider
    #    never sent one (zero-arg shape).
    (
        """            # Use a fallback item_id if not found
            if not matching_item_id:
                matching_item_id = f"tc_{uuid.uuid4()}"
""",
        """            # Use a fallback item_id if not found
            if not matching_item_id:
                matching_item_id = f"tc_{uuid.uuid4()}"
            # The accumulated entry's id may be None (the provider never sent one for
            # the zero-arg shape); backfill it so OpenAIToolMessageParam validates.
            if tool_call.id is None:
                tool_call.id = ""
""",
        'backfill it so OpenAIToolMessageParam validates',
    ),
    # 5. Hallucinated-tool fallback: tolerate id=None so the self-correcting
    #    error message reaches the model instead of a 500.
    (
        """                            next_turn_messages.append(
                                OpenAIToolMessageParam(
                                    tool_call_id=tool_call.id,
                                    content=(
                                        f"Error: tool '{tool_call.function.name}' is not available. "
""",
        """                            next_turn_messages.append(
                                OpenAIToolMessageParam(
                                    tool_call_id=tool_call.id or "",
                                    content=(
                                        f"Error: tool '{tool_call.function.name}' is not available. "
""",
        'tool_call_id=tool_call.id or "",\n                                    content=(',
    ),
    # 6. Skipped-call message: tolerate id=None.
    (
        """                skipped_call_message = OpenAIToolMessageParam(
                    content=f"Tool call skipped: maximum tool calls limit ({self.max_tool_calls}) reached.",
                    tool_call_id=tool_call.id,
                )
""",
        """                skipped_call_message = OpenAIToolMessageParam(
                    content=f"Tool call skipped: maximum tool calls limit ({self.max_tool_calls}) reached.",
                    tool_call_id=tool_call.id or "",
                )
""",
        'tool_call_id=tool_call.id or "",\n                )\n                next_turn_messages.append(skipped_call_message)',
    ),
]

TOOL_EXECUTOR_REPLACEMENTS: list[tuple[str, str, str]] = [
    # 1. MCP isError result: the error text is in content, not error_message.
    #    Only the error branch changes; a result with error_message keeps it,
    #    and a successful call never reaches this branch.
    (
        """                ec = getattr(result, "error_code", "unknown")
                em = getattr(result, "error_message", "")
                message.error = f"Error (code {ec}): {em}"
""",
        """                ec = getattr(result, "error_code", "unknown")
                em = getattr(result, "error_message", "")
                # The MCP runtime sets error_code and leaves error_message None; the
                # server's error text is in content. Keep it rather than render "None".
                if not em and (error_content := getattr(result, "content", None)):
                    em = interleaved_content_as_str(error_content)
                message.error = f"Error (code {ec}): {em}"
""",
        "em = interleaved_content_as_str(error_content)",
    ),
]

# File name under ogx/providers/inline/responses/builtin/responses/, with its
# replacements.
TARGETS: dict[str, list[tuple[str, str, str]]] = {
    "streaming.py": STREAMING_REPLACEMENTS,
    "tool_executor.py": TOOL_EXECUTOR_REPLACEMENTS,
}


def ogx_path(filename: str) -> Path | None:
    for base in Path(sys.prefix, "lib").glob("python*/site-packages"):
        path = (
            base
            / "ogx"
            / "providers"
            / "inline"
            / "responses"
            / "builtin"
            / "responses"
            / filename
        )
        if path.is_file():
            return path
    return None


def apply(
    text: str, replacements: list[tuple[str, str, str]], filename: str
) -> tuple[str, int, list[str]]:
    """Return the patched text, the count of replacements that landed, and one
    message for each replacement whose pre-patch text is gone or ambiguous."""
    patched = 0
    problems = []
    for index, (old, new, marker) in enumerate(replacements, start=1):
        if marker in text:
            continue
        matches = text.count(old)
        if matches == 1:
            text = text.replace(old, new)
            patched += 1
        elif matches == 0:
            problems.append(
                f"replacement {index}: ogx {filename} no longer matches the "
                f"pre-patch text (upstream changed)"
            )
        else:
            problems.append(
                f"replacement {index}: the pre-patch text matches {matches} "
                f"times in ogx {filename}; expected once"
            )
    return text, patched, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="change nothing; exit 1 unless every patch marker is present",
    )
    args = parser.parse_args(argv)

    sources = {}
    for filename in TARGETS:
        path = ogx_path(filename)
        if path is None:
            print(f"ogx {filename} not found; is ogx installed?", file=sys.stderr)
            return 1
        sources[filename] = (path, path.read_text())

    if args.check:
        failed = False
        for filename, replacements in TARGETS.items():
            text = sources[filename][1]
            missing = [
                index
                for index, (_, _, marker) in enumerate(replacements, start=1)
                if marker not in text
            ]
            found = len(replacements) - len(missing)
            print(f"ogx {filename}: {found}/{len(replacements)} markers")
            if missing:
                print(
                    f"ogx {filename} is missing patch replacements {missing}",
                    file=sys.stderr,
                )
                failed = True
        return 1 if failed else 0

    # Plan every file before writing any, so a drifted target leaves the
    # others as they were.
    plans = {}
    problems = []
    for filename, replacements in TARGETS.items():
        path, text = sources[filename]
        patched_text, patched, file_problems = apply(text, replacements, filename)
        plans[filename] = (path, patched_text, patched)
        problems.extend(file_problems)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print("no file written", file=sys.stderr)
        return 1

    for filename, (path, patched_text, patched) in plans.items():
        if patched:
            path.write_text(patched_text)
            print(f"patched {path} ({patched} replacement(s))")
        else:
            print(f"ogx {filename} already patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
