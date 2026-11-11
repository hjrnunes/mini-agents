"""scripts/patch_ogx_streaming.py patches two OGX sources.

The script tests run on a fake OGX tree built from the script's own pre-patch
blocks, so they need no OGX. The behaviour tests run the real OGX 1.3.1
``tool_executor.py``, patched in memory, on results the real MCP parser
produces; they skip where OGX is not installed (the ``gateway`` extra).
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESPONSES = ("ogx", "providers", "inline", "responses", "builtin", "responses")
FILENAMES = ["streaming.py", "tool_executor.py"]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "patch_ogx_streaming", ROOT / "scripts" / "patch_ogx_streaming.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script():
    return _load_script()


# Streaming replacement 6 has a marker that reads the upstream line after its
# block, so the fake source follows every block with that line.
FOLLOWING_LINE = "                next_turn_messages.append(skipped_call_message)\n"


def _pre_patch(script, filename):
    return "".join(old + FOLLOWING_LINE for old, _, _ in script.TARGETS[filename])


@pytest.fixture
def fake_ogx(tmp_path, monkeypatch, script):
    package = tmp_path / "lib" / "python3.12" / "site-packages" / Path(*RESPONSES)
    package.mkdir(parents=True)
    for filename in script.TARGETS:
        (package / filename).write_text(_pre_patch(script, filename))
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    return package


def _snapshot(package):
    return {name: (package / name).read_text() for name in FILENAMES}


def test_the_patch_lands_in_both_sources(script, fake_ogx):
    assert script.main([]) == 0

    for filename, replacements in script.TARGETS.items():
        text = (fake_ogx / filename).read_text()
        for _, new, marker in replacements:
            assert new in text
            assert marker in text
    assert script.main(["--check"]) == 0


def test_a_second_run_changes_nothing(script, fake_ogx, capsys):
    assert script.main([]) == 0
    patched = _snapshot(fake_ogx)
    capsys.readouterr()

    assert script.main([]) == 0

    assert _snapshot(fake_ogx) == patched
    assert capsys.readouterr().out.count("already patched") == len(FILENAMES)


def test_a_run_after_a_partial_patch_finishes_it(script, fake_ogx, capsys):
    streaming = fake_ogx / "streaming.py"
    text = streaming.read_text()
    for old, new, _ in script.STREAMING_REPLACEMENTS[:-1]:
        text = text.replace(old, new)
    streaming.write_text(text)

    assert script.main([]) == 0

    assert "streaming.py (1 replacement(s))" in capsys.readouterr().out
    assert script.main(["--check"]) == 0


def test_check_fails_on_an_unpatched_source_and_changes_nothing(
    script, fake_ogx, capsys
):
    before = _snapshot(fake_ogx)

    assert script.main(["--check"]) == 1

    assert _snapshot(fake_ogx) == before
    captured = capsys.readouterr()
    assert "ogx tool_executor.py: 0/1 markers" in captured.out
    assert "ogx tool_executor.py is missing patch replacements [1]" in captured.err


@pytest.mark.parametrize("filename", FILENAMES)
def test_a_changed_source_fails_clearly_and_writes_nothing(
    script, fake_ogx, capsys, filename
):
    old = script.TARGETS[filename][0][0]
    drifted = (fake_ogx / filename).read_text().replace(old, "# rewritten upstream\n")
    (fake_ogx / filename).write_text(drifted)
    before = _snapshot(fake_ogx)

    assert script.main([]) == 1

    # Nothing is written, not even the file that still matches.
    assert _snapshot(fake_ogx) == before
    err = capsys.readouterr().err
    assert f"replacement 1: ogx {filename} no longer matches the pre-patch text" in err
    assert "no file written" in err


def test_a_pre_patch_block_that_matches_twice_fails(script, fake_ogx, capsys):
    old = script.TOOL_EXECUTOR_REPLACEMENTS[0][0]
    (fake_ogx / "tool_executor.py").write_text(old + "\n" + old)
    before = _snapshot(fake_ogx)

    assert script.main([]) == 1

    assert _snapshot(fake_ogx) == before
    assert "matches 2 times in ogx tool_executor.py" in capsys.readouterr().err


@pytest.mark.parametrize("filename", FILENAMES)
def test_a_marker_without_the_patched_text_fails_clearly(
    script, fake_ogx, capsys, filename
):
    """The defective block is back and a comment keeps the marker."""
    assert script.main([]) == 0
    old, new, marker = script.TARGETS[filename][0]
    text = (fake_ogx / filename).read_text().replace(new, old + f"# {marker}\n")
    (fake_ogx / filename).write_text(text)
    before = _snapshot(fake_ogx)
    capsys.readouterr()

    assert script.main([]) == 1

    assert _snapshot(fake_ogx) == before
    err = capsys.readouterr().err
    assert (
        f"replacement 1: ogx {filename} has the patch marker but not the patched text"
        in err
    )
    assert script.main(["--check"]) == 1
    assert f"ogx {filename} is missing patch replacements [1]" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [[], ["--check"]])
def test_a_missing_ogx_fails(script, tmp_path, monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, "prefix", str(tmp_path))

    assert script.main(argv) == 1

    assert "not found; is ogx installed?" in capsys.readouterr().err


# The installed OGX.


requires_ogx = pytest.mark.skipif(
    importlib.util.find_spec("ogx") is None,
    reason="OGX is not installed (the gateway extra)",
)


def _installed_source(filename):
    spec = importlib.util.find_spec("ogx")
    assert spec is not None and spec.submodule_search_locations
    path = Path(spec.submodule_search_locations[0]).joinpath(*RESPONSES[1:], filename)
    return path, path.read_text()


@requires_ogx
@pytest.mark.parametrize("filename", FILENAMES)
def test_the_installed_ogx_matches_the_patch(script, filename):
    """The pinned OGX still holds each pre-patch block once, or is patched."""
    _, text = _installed_source(filename)

    _, _, problems = script.apply(text, script.TARGETS[filename], filename)

    assert problems == []


@requires_ogx
def test_the_installed_executor_with_only_the_marker_is_refused(script):
    """The defect is back in the real source and a comment keeps the marker."""
    _, source = _installed_source("tool_executor.py")
    old, new, marker = script.TOOL_EXECUTOR_REPLACEMENTS[0]
    doctored = source.replace(new, old).replace(
        old, old + f"                # {marker}\n"
    )

    _, patched, problems = script.apply(
        doctored, script.TOOL_EXECUTOR_REPLACEMENTS, "tool_executor.py"
    )

    assert patched == 0
    assert len(problems) == 1


def _tool_executor(script, *, patched):
    """The installed tool_executor.py, with the isError replacement in or out."""
    path, source = _installed_source("tool_executor.py")
    replacements = script.TOOL_EXECUTOR_REPLACEMENTS
    for old, new, _ in replacements:
        source = source.replace(new, old)
    if patched:
        source, _, problems = script.apply(source, replacements, "tool_executor.py")
        assert problems == []
    package = ".".join(RESPONSES)
    module = types.ModuleType(f"{package}._tool_executor_test")
    module.__package__ = package
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def _mcp_result(*texts, is_error):
    """What OGX's MCP runtime makes of a server's tools/call result."""
    from mcp import types as mcp_types

    from ogx.providers.utils.tools.mcp import _parse_mcp_result

    return _parse_mcp_result(
        mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text=t) for t in texts],
            isError=is_error,
        )
    )


def _build(module, result, error_exc=None):
    """The MCP call item and the tool message for the model."""
    from ogx_api import OpenAIResponseInputToolMCP

    name = "modify_booking"
    executor = module.ToolExecutor.__new__(module.ToolExecutor)
    mcp_tools = {
        name: OpenAIResponseInputToolMCP(
            server_label="airbnb-safe", server_url="http://localhost:8890/sse"
        )
    }
    has_error = bool(
        error_exc
        or (result and ((result.error_code or 0) > 0 or result.error_message))
    )
    return asyncio.run(
        executor._build_result_messages(
            SimpleNamespace(name=name, arguments="{}"),
            "call_1",
            "item_1",
            {},
            None,
            error_exc,
            result,
            has_error,
            mcp_tools,
        )
    )


BOOM = "Error executing tool modify_booking: boom"


@requires_ogx
def test_an_is_error_result_keeps_the_mcp_text(script):
    result = _mcp_result(BOOM, is_error=True)

    message, tool_message = _build(_tool_executor(script, patched=True), result)

    assert message.error == f"Error (code 1): {BOOM}"
    assert message.output is None
    assert BOOM in str(tool_message.content)


@requires_ogx
def test_every_text_item_of_an_is_error_result_is_kept(script):
    result = _mcp_result("first line", "second line", is_error=True)

    message, _ = _build(_tool_executor(script, patched=True), result)

    assert "first line" in message.error and "second line" in message.error


@requires_ogx
def test_the_unpatched_executor_loses_the_text(script):
    """The control: the defect the patch fixes."""
    result = _mcp_result(BOOM, is_error=True)

    message, _ = _build(_tool_executor(script, patched=False), result)

    assert message.error == "Error (code 1): None"


def _plain_result(content, **fields):
    from ogx_api import TextContentItem, ToolInvocationResult

    items = [TextContentItem(text=text) for text in content]
    return ToolInvocationResult(content=items, **fields)


# Each case builds a result that must read the same patched and unpatched.
UNCHANGED = {
    "success": lambda: (_mcp_result('{"status": "ok"}', is_error=False), None),
    "success with empty content": lambda: (_mcp_result(is_error=False), None),
    "error message set": lambda: (
        _plain_result(["ignored"], error_code=2, error_message="denied"),
        None,
    ),
    "error with no content": lambda: (_mcp_result(is_error=True), None),
    "transport failure": lambda: (None, RuntimeError("transport down")),
}


@requires_ogx
@pytest.mark.parametrize("case", UNCHANGED)
def test_every_other_result_reads_the_same(script, case):
    result, error_exc = UNCHANGED[case]()

    unpatched = _build(_tool_executor(script, patched=False), result, error_exc)
    patched = _build(_tool_executor(script, patched=True), result, error_exc)

    assert patched[0].model_dump() == unpatched[0].model_dump()
    assert patched[1].model_dump() == unpatched[1].model_dump()


@requires_ogx
def test_the_tool_message_for_the_model_is_unchanged_on_error(script):
    result = _mcp_result(BOOM, is_error=True)

    unpatched = _build(_tool_executor(script, patched=False), result)
    patched = _build(_tool_executor(script, patched=True), result)

    assert patched[1].model_dump() == unpatched[1].model_dump()


@requires_ogx
def test_a_successful_call_still_reports_its_output_and_no_error(script):
    result = _mcp_result('{"status": "ok"}', is_error=False)

    message, _ = _build(_tool_executor(script, patched=True), result)

    assert message.error is None
    assert message.output == '{"status": "ok"}'
