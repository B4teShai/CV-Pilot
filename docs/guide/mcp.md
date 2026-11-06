# MCP server

`cvpilot mcp` serves CV-Pilot's run store over the
[Model Context Protocol](https://modelcontextprotocol.io) on standard input and output, so a
coding agent or any other MCP client can read runs, compare them with baselines and gate them
without shell access. It uses the same settings as the command line (`CVPILOT_STORE_DIR`,
`CVPILOT_PROJECT`) and the same fail-closed services.

```bash notest
uv sync --extra mcp              # in a clone; the extra holds the MIT-licensed `mcp` SDK
cvpilot mcp                      # started by the MCP client, not by hand
```

A client configuration (the file name and layout depend on the client):

```json
{
  "mcpServers": {
    "cvpilot": {
      "command": "cvpilot",
      "args": ["mcp"],
      "env": { "CVPILOT_PROJECT": "detector", "CVPILOT_MCP_ROOTS": "/data/checkpoints" }
    }
  }
}
```

## Tools

| Tool | Writes | What it returns |
|---|---|---|
| `runs_list` | no | Runs of a project, newest first. |
| `run_show` | no | Status, reproducibility record, metric counts, artifacts and evaluation snapshots. |
| `baselines_list`, `baseline_show` | no | Pinned baselines; `show` verifies the pin and its hash chain. |
| `compare` | no | Headline deltas, the worst classes and slices, confusion change counts. No verdict. |
| `gate` | no | Verdict, violations and the comparison. `max_drop` and the other thresholds use the command-line syntax (`mAP50_95=0.5`, `camera=*:1.5`); `gt` and `classes` re-evaluate stored predictions. |
| `ckpt_verify` | no | `cvpilot ckpt verify --json`, run in a child process: rows, and `ok` only when all verify. |
| `alerts_check` | no | `cvpilot alerts check` output; nothing is sent. |
| `baseline_set` | pins a tag | The new pin. Re-pinning keeps the history. |

## Safety rules

- **Read-only by default.** `baseline_set` exists only when the server starts with
  `CVPILOT_MCP_ALLOW_WRITE=1`. No tool deletes anything.
- **Path roots.** Checkpoint directories, alert configs, ground truth and class files must resolve,
  symlinks followed, inside `CVPILOT_MCP_ROOTS` (separated by `:`, or `;` on Windows; default: the
  directory the server started in). The run store is the server's own setting, not an argument.
- **Bounded output.** Lists stop at `limit` (at most 100); classes and slices are the worst `top_k`
  (at most 50); confusion matrices are counted, not listed.
- **Errors** come back as tool errors with a one-line reason. Logs are the usual JSON records on
  standard error, one `request_id` per tool call; an unexpected failure is logged once, with its
  traceback, and the client only sees its type.
