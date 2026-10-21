#!/usr/bin/env bash
# Smoke-check the sandbox image built from Dockerfile.sandbox. Every check runs
# in its own container with no network. Usage:
#   scripts/smoke_sandbox_image.sh [image]   (default asago-mini-agents-sandbox:dev)
set -euo pipefail

IMAGE="${1:-asago-mini-agents-sandbox:dev}"
# Seconds a domain server gets to serve /sse; kernel/stack.py waits 20.
READY_BUDGET=20
failures=0

check() {
    local name="$1"
    shift
    local output
    if output="$(docker run --rm --network none "$IMAGE" "$@" 2>&1)"; then
        echo "PASS  $name"
    else
        echo "FAIL  $name"
        printf '%s\n' "$output" | sed 's/^/      /'
        failures=$((failures + 1))
    fi
}

echo "image: $IMAGE"

check "non-root user (id -u != 0)" \
    sh -c 'uid="$(id -u)"; echo "uid=$uid"; [ "$uid" != 0 ]'

check "tiktoken cl100k_base loads offline" \
    python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

check "import mini_agents" \
    python -c "import mini_agents"

check "klarna MCP server serves /sse on loopback within ${READY_BUDGET}s" \
    bash -c '
        mini-agents --domain klarna --host 127.0.0.1 --port 8888 >/tmp/server.log 2>&1 &
        deadline=$((SECONDS + '"$READY_BUDGET"'))
        while [ "$SECONDS" -lt "$deadline" ]; do
            # /sse streams until closed, so curl times out; the status code is enough.
            code="$(curl -s -o /dev/null -w "%{http_code}" --max-time 1 http://127.0.0.1:8888/sse || true)"
            if [ "$code" = 200 ]; then
                echo "ready after ${SECONDS}s"
                exit 0
            fi
            sleep 0.2
        done
        cat /tmp/server.log
        exit 1
    '

check "ogx --help" \
    ogx --help

check "OGX streaming patch present (all replacements)" \
    python -c "
import sys
sys.path.insert(0, 'scripts')
import patch_ogx_streaming as p
text = p.streaming_path().read_text()
missing = [i for i, (_, _, marker) in enumerate(p.REPLACEMENTS, 1) if marker not in text]
print(f'{len(p.REPLACEMENTS) - len(missing)}/{len(p.REPLACEMENTS)} markers')
sys.exit(1 if missing else 0)
"

check "console scripts on PATH" \
    sh -c 'for c in python ogx mini-agents mini-agents-stack; do command -v "$c" || exit 1; done'

check "writable \$HOME and /tmp" \
    sh -c 'touch "$HOME/.probe" /tmp/.probe && echo "HOME=$HOME"'

if [ "$failures" -ne 0 ]; then
    echo "$failures check(s) failed"
    exit 1
fi
echo "all checks passed"
