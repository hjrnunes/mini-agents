"""Patch OGX's streaming tool-call merge for vLLM Hermes-style deltas.

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

Each replacement carries a marker substring that appears only in the patched
text, so the script is idempotent and safe to re-run after partial upgrades.
"""

from __future__ import annotations

import sys
from pathlib import Path

# (old, new, marker) — marker is a substring unique to the patched text.
REPLACEMENTS: list[tuple[str, str, str]] = [
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


def streaming_path() -> Path | None:
    for base in Path(sys.prefix, "lib").glob("python*/site-packages"):
        path = (
            base
            / "ogx"
            / "providers"
            / "inline"
            / "responses"
            / "builtin"
            / "responses"
            / "streaming.py"
        )
        if path.is_file():
            return path
    return None


def main() -> int:
    streaming = streaming_path()
    if streaming is None:
        print("ogx streaming.py not found; is ogx installed?", file=sys.stderr)
        return 1
    text = streaming.read_text()
    patched = 0
    for index, (old, new, marker) in enumerate(REPLACEMENTS, start=1):
        if marker in text:
            continue
        if old in text:
            text = text.replace(old, new)
            patched += 1
        else:
            print(
                f"replacement {index}: ogx streaming.py no longer matches the "
                f"pre-patch text; skipping (upstream may have fixed it)",
                file=sys.stderr,
            )
    if patched:
        streaming.write_text(text)
        print(f"patched {streaming} ({patched} replacement(s))")
    else:
        print("ogx streaming.py already patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
