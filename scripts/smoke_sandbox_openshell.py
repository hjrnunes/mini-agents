"""Smoke-check the sandbox image and its policy under a running OpenShell gateway.

Run it with an interpreter that has the OpenShell SDK (MiDojo's venv), against
the bare image ID of an image built from Dockerfile.sandbox:

    <midojo venv>/bin/python scripts/smoke_sandbox_openshell.py sha256:<id>

It starts two fake HTTP servers on host loopback (the control plane and the
model relay), creates one sandbox from the image with
src/mini_agents/sandbox-policy.json, its port slots filled with those servers'
ports, and checks identity, writable paths, egress per interpreter copy and
offline tiktoken. It prints PASS or FAIL per check and exits 1 on any failure.
The sandbox and its workspace are deleted on the way out. No model is called.
README "Sandbox image".
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import socketserver
import sys
import threading
import time
import uuid
from pathlib import Path

import grpc
from google.protobuf.json_format import ParseDict
from openshell import SandboxClient, WorkspaceClient
from openshell._proto import openshell_pb2

ROOT = Path(__file__).resolve().parent.parent
HOST = "host.openshell.internal"
VBIN = "/opt/app-root/.venv/bin"
WORKDIR = "/sandbox/workdir"
LABELS = {"mini-agents.smoke": "openshell"}
IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
# The probe each interpreter runs: GET / on a host port, print what came back.
PROBE = """
import sys, urllib.request
try:
    body = urllib.request.urlopen(f"http://{sys.argv[1]}:{sys.argv[2]}/", timeout=5).read()
    print("reached", body.decode().strip())
except Exception as exc:
    print("blocked", type(exc).__name__, exc)
"""


def _sandbox_policy():
    # Load the module by path: this interpreter does not have mini_agents.
    path = ROOT / "src" / "mini_agents" / "kernel" / "sandbox_policy.py"
    spec = importlib.util.spec_from_file_location("sandbox_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Answer(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            self.rfile.readline()
        except OSError:
            pass
        body = f"fake-{self.server.role}-{self.server.server_address[1]}\n".encode()
        self.wfile.write(
            b"HTTP/1.0 200 OK\r\nContent-Length: %d\r\n\r\n" % len(body) + body
        )


class _FakeServer(socketserver.ThreadingTCPServer):
    daemon_threads = True

    def __init__(self, role):
        super().__init__(("127.0.0.1", 0), _Answer)
        self.role = role
        self.port = self.server_address[1]
        self.marker = f"fake-{role}-{self.port}"
        threading.Thread(target=self.serve_forever, daemon=True).start()

    def close(self):
        self.shutdown()
        self.server_close()


def _not_found(exc):
    code = getattr(exc, "code", None)
    return callable(code) and code() == grpc.StatusCode.NOT_FOUND


class Smoke:
    def __init__(self):
        self.client = SandboxClient.from_active_cluster(cluster="openshell", timeout=120.0)
        self.workspaces = WorkspaceClient.from_sandbox_client(self.client)
        # Both names are fixed before any create call, so close() can delete
        # them even when a create succeeds on the gateway but fails here.
        # OpenShell caps workspace names at 19 characters.
        self.workspace = f"ma-smoke-{uuid.uuid4().hex[:8]}"
        self.sandbox = "smoke"
        self.attempted = False
        self.failures = 0

    def create(self, image, policy):
        spec = openshell_pb2.SandboxSpec(
            template=openshell_pb2.SandboxTemplate(image=image)
        )
        ParseDict(policy, spec.policy)
        self.attempted = True
        print(f"workspace {self.workspace}", flush=True)
        self.workspaces.create(self.workspace, labels=LABELS)
        self.client.create(
            workspace=self.workspace, spec=spec, name=self.sandbox, labels=LABELS
        )
        started = time.time()
        self.client.wait_ready(
            self.sandbox, workspace=self.workspace, timeout_seconds=180.0
        )
        print(f"sandbox {self.sandbox} ready in {time.time() - started:.1f}s", flush=True)

    def run(self, command, timeout=60):
        result = self.client.exec(
            self.sandbox,
            command,
            workspace=self.workspace,
            workdir=WORKDIR,
            timeout_seconds=timeout,
        )
        return result.exit_code, result.stdout.strip(), result.stderr.strip()

    def check(self, name, command, ok):
        """Run ``command``; ``ok(exit_code, stdout)`` decides PASS."""
        try:
            code, out, err = self.run(command)
            passed = ok(code, out)
        except Exception as exc:  # an exec that cannot start is a failure
            code, out, err, passed = None, "", f"{type(exc).__name__}: {exc}", False
        print(f"{'PASS' if passed else 'FAIL'}  {name}", flush=True)
        if not passed:
            self.failures += 1
            for line in [f"exit={code}", *out.splitlines(), *err.splitlines()[-10:]]:
                print(f"      {line}", flush=True)
        return out

    def _delete_sandbox(self, name):
        result = self.client.delete(name, workspace=self.workspace, allow_missing=True)
        self.client.wait_deleted(name, workspace=self.workspace, timeout_seconds=120)
        print(f"deleted sandbox {name} ({result.outcome.name})", flush=True)

    def _sandboxes_left(self):
        return [
            sandbox.name
            for page in self.client.list(workspace=self.workspace)
            for sandbox in (page.items if hasattr(page, "items") else [page])
        ]

    def close(self):
        """Delete the sandbox, anything else in the workspace, then the workspace.

        Each step tolerates names that do not exist; a missing workspace means
        nothing was created in it.
        """
        if not self.attempted:
            return
        try:
            self._delete_sandbox(self.sandbox)
            for _ in range(60):
                left = self._sandboxes_left()
                if not left:
                    break
                for name in left:
                    self._delete_sandbox(name)
                time.sleep(1)
            else:
                raise RuntimeError(f"sandboxes still in the workspace: {left}")
        except Exception as exc:
            if not _not_found(exc):
                print(f"sandbox delete failed: {exc}", flush=True)
                self.failures += 1
        try:
            result = self.workspaces.delete(self.workspace, allow_missing=True)
            print(f"deleted workspace {self.workspace} ({result.outcome.name})", flush=True)
        except Exception as exc:
            print(f"workspace delete failed: {exc}", flush=True)
            self.failures += 1


def _probe_checks(smoke, interpreter, allowed, denied):
    label = interpreter.rsplit("/", 1)[1]
    for server, should_reach in ((allowed, True), (denied, False)):
        if server is None:
            continue
        verb = "reaches" if should_reach else "is denied"
        smoke.check(
            f"{label} {verb} the {server.role} port ({server.port})",
            [interpreter, "-c", PROBE, HOST, str(server.port)],
            lambda code, out, server=server, should_reach=should_reach: code == 0
            and (out == f"reached {server.marker}") == should_reach
            and (should_reach or out.startswith("blocked")),
        )


def _checks(smoke, control, relay):
    # MiDojo creates the workdir before it runs anything there.
    smoke.client.exec(
        smoke.sandbox, ["mkdir", "-p", WORKDIR], workspace=smoke.workspace,
        timeout_seconds=30,
    )

    smoke.check("uid 1500", ["id", "-u"], lambda c, o: c == 0 and o == "1500")
    smoke.check(
        "HOME=/sandbox", ["sh", "-c", 'echo "$HOME"'],
        lambda c, o: c == 0 and o == "/sandbox",
    )
    smoke.check("cwd /sandbox/workdir", ["pwd"], lambda c, o: c == 0 and o == WORKDIR)
    for path in (WORKDIR, "/tmp", "/dev/shm"):
        smoke.check(
            f"{path} is writable",
            ["sh", "-c", f'p="{path}/.smoke-$$" && touch "$p" && rm "$p"'],
            lambda c, o: c == 0,
        )
    # What OGX's job workers create at startup; fails on a read-only /dev/shm.
    smoke.check(
        "python-ogx creates a multiprocessing semaphore",
        [f"{VBIN}/python-ogx", "-c",
         "import multiprocessing as m; m.get_context('spawn').Event(); print('ok')"],
        lambda c, o: c == 0 and o == "ok",
    )

    _probe_checks(smoke, f"{VBIN}/python-mcp", allowed=control, denied=relay)
    _probe_checks(smoke, f"{VBIN}/python-ogx", allowed=relay, denied=control)
    for server in (control, relay):
        _probe_checks(smoke, f"{VBIN}/python", allowed=None, denied=server)

    smoke.check(
        "tiktoken cl100k_base loads offline with TIKTOKEN_CACHE_DIR set",
        ["env", "TIKTOKEN_CACHE_DIR=/opt/app-root/tiktoken", f"{VBIN}/python", "-c",
         "import tiktoken; print(len(tiktoken.get_encoding('cl100k_base').encode('hello world')))"],
        lambda c, o: c == 0 and o == "2",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("image", help="bare image ID, sha256:<64 hex digits>")
    args = parser.parse_args(argv)
    if not IMAGE_ID.match(args.image):
        parser.error(
            "pass the bare image ID, sha256:<64 hex digits> "
            "(docker image inspect <tag> --format '{{.Id}}'); "
            "a tag can drift and name@sha256: makes OpenShell try a pull"
        )
    print(f"image {args.image}")

    servers = []
    try:
        control = _FakeServer("control-plane")
        servers.append(control)
        relay = _FakeServer("relay")
        servers.append(relay)
        policy = _sandbox_policy().fill_policy(
            control_plane_port=control.port, relay_port=relay.port
        )
        print(f"control-plane port {control.port}, relay port {relay.port}", flush=True)

        smoke = Smoke()
        try:
            smoke.create(args.image, policy)
            _checks(smoke, control, relay)
        except Exception as exc:
            print(f"FAIL  sandbox setup: {type(exc).__name__}: {exc}", flush=True)
            smoke.failures += 1
        finally:
            smoke.close()
    finally:
        for server in servers:
            server.close()

    if smoke.failures:
        print(f"{smoke.failures} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
