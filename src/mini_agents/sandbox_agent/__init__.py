"""The in-sandbox agent command, ``mini-agents-sandbox-agent``.

MiDojo's openshell runtime execs it once per turn inside the sandbox image.
The first turn starts the domain's safe MCP server in MiDojo mode and OGX,
both detached and bound to loopback; every turn sends one Responses request
to OGX and prints only the reply on stdout. Turn state lives in
``/tmp/agent-state``, evidence in ``/sandbox/workdir/.asago``. README "Sandbox
agent command" has the inputs and the files.

Parts: ``ogx_config`` renders OGX's config, ``state`` keeps the turn state,
``responses`` builds the request, ``processes`` starts and checks the servers,
``environment`` builds their environments, ``summary`` writes the evidence,
and ``cli`` runs one turn.
"""
