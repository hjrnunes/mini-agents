"""A stand-in for OGX in the gateway shutdown tests.

    python stub_ogx.py CONFIG READY

``CONFIG`` is the OGX config the gateway rendered. The stand-in listens on the
port that config names, as OGX does, writes ``READY`` (its pid, that port and
the shim URL the config names) and then behaves as ``STUB_OGX_MODE`` says:

- ``ignore-sigterm`` runs until it is killed and ignores SIGTERM.
- ``exit-on-sigterm`` runs until SIGTERM, which ends it with the default action.
- ``exit:N`` exits with code N a moment after it is ready.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import time
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text())
ready = Path(sys.argv[2])
mode = os.environ["STUB_OGX_MODE"]

port = int(config["server"]["port"])
shim_url = config["providers"]["inference"][0]["config"]["base_url"]

if mode == "ignore-sigterm":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", port))
listener.listen()

partial = ready.with_suffix(".partial")
partial.write_text(json.dumps({"pid": os.getpid(), "port": port, "shim_url": shim_url}))
partial.rename(ready)

if mode.startswith("exit:"):
    time.sleep(0.3)
    sys.exit(int(mode.split(":", 1)[1]))
while True:
    time.sleep(0.2)
