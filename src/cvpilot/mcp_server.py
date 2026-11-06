"""MCP server: recorded runs, baselines, comparisons, gates and checkpoint checks as agent tools.

Start it with ``cvpilot mcp`` (extra ``cvpilot[mcp]``). It is an interface layer like the command
line: the run store comes from the usual settings (``CVPILOT_STORE_DIR``, ``CVPILOT_PROJECT``), and
the same services do the work with the same fail-closed rules.

Safety rules, enforced here and not left to the caller:

* **Read-only by default.** The only tool that writes, ``baseline_set`` (pins a baseline, never
  deletes one), is registered only when ``CVPILOT_MCP_ALLOW_WRITE=1``.
* **Path roots.** Path arguments (checkpoint directory, alert config, ground truth, class names)
  must resolve, symlinks followed, inside ``CVPILOT_MCP_ROOTS`` (``os.pathsep``-separated;
  default: the working directory at start). Relative paths are taken from the first root.
* **Bounded output.** Lists stop at ``limit`` (100 at most); per-class and slice deltas are the
  worst ``top_k``; confusion matrices are counted, not listed.
* **Logging.** Each call gets its own ``request_id``; JSON records go to standard error (standard
  output is the protocol). A failure is logged once, here.

``ckpt_verify`` and ``alerts_check`` run the command line's own ``--json``/text output in a child
process (the same interpreter, isolated mode) so that their trust policy is exactly the CLI's.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field

from cvpilot.core.errors import CvpilotError
from cvpilot.core.text import safe_text
from cvpilot.logging import clear_request_context, configure_logging, get_logger, scrub
from cvpilot.settings import Settings, load_settings

if TYPE_CHECKING:  # the mcp package is the extra cvpilot[mcp]: imported when serving only
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

    from cvpilot.compare.engine import Comparison

__all__ = ["ROOTS_ENV", "WRITE_ENV", "create_server", "main"]

WRITE_ENV = "CVPILOT_MCP_ALLOW_WRITE"
ROOTS_ENV = "CVPILOT_MCP_ROOTS"
MAX_LIMIT = 100
MAX_TOP_K = 50
CHILD_TIMEOUT_S = 900
_LINES = 80

RunId = Annotated[str, Field(pattern=r"^[0-9A-Za-z]{26}$", description="Run id (ULID).")]
Project = Annotated[
    str | None, Field(description="Project name; default: CVPILOT_PROJECT of the server.")
]
Baseline = Annotated[
    str, Field(description="Baseline: 'tag:<name>' for a pinned baseline, or a run id.")
]
EvalSet = Annotated[
    str | None, Field(description="Eval set hash or prefix (8+ hex); optional with one eval set.")
]
Step = Annotated[int | None, Field(ge=0, description="Snapshot step; default: the latest.")]
TopK = Annotated[int, Field(ge=1, le=MAX_TOP_K, description="Worst classes and slices to return.")]
Limit = Annotated[int, Field(ge=1, le=MAX_LIMIT)]


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _env_roots() -> tuple[Path, ...]:
    raw = os.environ.get(ROOTS_ENV, "")
    parts = [p for p in raw.split(os.pathsep) if p.strip()] or [os.getcwd()]
    return tuple(Path(p).expanduser().resolve() for p in parts)


class Refused(ValueError):
    """A tool argument the server will not act on (a path outside the roots, no project)."""


class Paths:
    """Resolve tool path arguments and refuse anything outside the allowed roots."""

    def __init__(self, roots: tuple[Path, ...]) -> None:
        if not roots:
            raise ValueError("at least one root is required")
        self.roots = roots

    def existing(self, value: str, *, directory: bool = False) -> Path:
        if not value or "\x00" in value:
            raise Refused("empty or invalid path")
        path = Path(value).expanduser()
        resolved = (path if path.is_absolute() else self.roots[0] / path).resolve()
        if not any(resolved == r or r in resolved.parents for r in self.roots):
            raise Refused(f"{value!r} is outside the allowed roots (set {ROOTS_ENV})")
        if not resolved.exists():
            raise Refused(f"{value!r} does not exist")
        if directory != resolved.is_dir():
            raise Refused(f"{value!r} is {'not ' if directory else ''}a directory")
        return resolved


@contextmanager
def _call(tool: str) -> Iterator[Settings]:
    """One tool call: a fresh request id, library errors as tool errors, one log on failure."""
    from mcp.server.mcpserver.exceptions import ToolError

    settings = load_settings()
    configure_logging("cvpilot", settings)
    log = get_logger("cvpilot.mcp")
    try:
        yield settings
    except (CvpilotError, ValueError) as exc:  # refused input or data: the caller can fix it
        log.warning("tool_refused", tool=tool, error_type=type(exc).__name__)
        text = scrub(safe_text(str(exc), 600))
        raise ToolError(
            text if isinstance(exc, Refused) else f"{type(exc).__name__}: {text}"
        ) from None
    except OSError as exc:  # says what failed, not where: a path can carry a user name
        log.error("tool_failed", tool=tool, exc_info=exc)
        raise ToolError(f"OSError: {exc.strerror or type(exc).__name__}") from None
    except Exception as exc:  # error boundary: logged once here, no internals to the caller
        log.error("tool_failed", tool=tool, exc_info=exc)
        raise ToolError(f"internal error ({type(exc).__name__}); see the server log") from None
    finally:
        clear_request_context()


def _project(settings: Settings, project: str | None) -> str:
    from cvpilot.core.schema import PROJECT_PATTERN, is_valid_project

    name = project or settings.project
    if name is None:
        raise Refused("no project: pass project or start the server with CVPILOT_PROJECT")
    if not is_valid_project(name):
        raise Refused(f"project must match {PROJECT_PATTERN}")
    return name


def _dump(model: Any) -> Any:
    return json.loads(model.model_dump_json())


def _worst(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Rows with the largest drop first; rows without a delta (missing, new) right after."""

    def key(row: dict[str, Any]) -> tuple[int, float]:
        delta = row.get("delta_points")
        return (0, float(delta)) if isinstance(delta, (int, float)) else (1, 0.0)

    return sorted(rows, key=key)[:top_k]


def comparison_summary(cmp: Comparison, top_k: int) -> dict[str, Any]:
    """Bounded view of a comparison: refs, every headline delta, the worst classes and slices."""
    from cvpilot.compare.report import report_dict

    return report_summary(report_dict(cmp), top_k)


def report_summary(full: dict[str, Any], top_k: int) -> dict[str, Any]:
    """The same bounded view from a ``--json`` report of ``cvpilot compare`` or ``gate``."""
    per_class = [row for rows in full["per_class"].values() for row in rows]
    confusion = full.get("confusion") or {}
    return {
        "unit": full["unit"],
        "candidate": full["candidate"],
        "baseline": full["baseline"],
        "headline": full["headline"],
        "worst_classes": _worst(per_class, top_k),
        "class_rows": len(per_class),
        "worst_slices": _worst(list(full["slices"]), top_k),
        "slice_rows": len(full["slices"]),
        "confusion": {k: v for k, v in confusion.items() if k != "cells"} or None,
    }


def _cli(args: list[str], settings: Settings) -> tuple[int, str, str]:
    """Run ``cvpilot <args>`` in a child process of this interpreter (isolated mode)."""
    code = "from cvpilot.cli.app import main; main()"
    done = subprocess.run(  # noqa: S603  # fixed interpreter and module, list arguments, no shell
        [sys.executable, "-I", "-c", code, *args],
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_S,
        check=False,
        env={**os.environ, "CVPILOT_LOG_LEVEL": "WARNING"},
    )
    return done.returncode, done.stdout, done.stderr


def _gate_recompute(
    settings: Settings,
    project: str,
    run_id: str,
    baseline: str,
    gt: Path,
    classes: Path,
    max_drop: list[str],
    per_class_max_drop: list[str] | None,
    slice_max_drop: list[str] | None,
    min_images: int,
    skip_small_slices: bool,
    max_skipped: int,
    eval_set: str | None,
    step: int | None,
    top_k: int,
) -> dict[str, Any]:
    """``cvpilot gate --recompute --json`` in a child process; exit 1 is a verdict, 2 a refusal."""
    args = ["gate", run_id, "--project", project, "--baseline", baseline, "--recompute",
            "--gt", str(gt), "--classes", str(classes), "--min-images", str(min_images),
            "--max-skipped", str(max_skipped), "--json"]  # fmt: skip
    for flag, values in (("--max-drop", max_drop), ("--per-class-max-drop", per_class_max_drop),
                         ("--slice-max-drop", slice_max_drop)):  # fmt: skip
        for value in values or []:
            args += [flag, value]
    if skip_small_slices:
        args.append("--skip-small-slices")
    if eval_set:
        args += ["--eval-set", eval_set]
    if step is not None:
        args += ["--step", str(step)]
    code, out, err = _cli(args, settings)
    report = None
    for line in reversed(out.splitlines()):
        if line.startswith("{"):
            report = json.loads(line)
            break
    if code not in (0, 1) or report is None or report.get("gate") is None:
        reason = err.strip().splitlines()[-1] if err.strip() else f"exit {code}"
        raise Refused(scrub(safe_text(reason, 300)))
    verdict = report["gate"]
    verdict["violations"] = verdict["violations"][:MAX_LIMIT]
    return {"passed": bool(verdict["passed"]), "gate": verdict,
            "comparison": report_summary(report, top_k)}  # fmt: skip


def _register_read(server: MCPServer, paths: Paths, read: ToolAnnotations) -> None:
    @server.tool(title="List runs", annotations=read)
    def runs_list(
        project: Project = None,
        status: Literal["running", "resumed", "finished", "failed", "preempted"] | None = None,
        limit: Limit = 20,
    ) -> dict[str, Any]:
        """List recorded runs of a project, newest first: id, name, status, times, task."""
        from cvpilot import runs

        with _call("runs_list") as settings:
            found = runs.list_runs(
                _project(settings, project), status=status, store_dir=settings.store_dir
            )
            rows = [
                {
                    "run_id": r.run_id,
                    "name": r.name,
                    "status": r.status,
                    "task": r.task,
                    "framework": r.framework,
                    "started_at": r.started_at.isoformat(),
                    "ended_at": r.ended_at.isoformat() if r.ended_at else None,
                    "parent_run_id": r.parent_run_id,
                }
                for r in found[:limit]
            ]
            return {"runs": rows, "total": len(found), "truncated": len(found) > limit}

    @server.tool(title="Show a run", annotations=read)
    def run_show(run_id: RunId, project: Project = None, limit: Limit = 20) -> dict[str, Any]:
        """Show one run: status, reproducibility record (commit, dataset hash, seed, GPUs), metric
        counts, artifacts and its evaluation snapshots (newest ``limit``) with headline metrics."""
        from cvpilot import runs

        with _call("run_show") as settings:
            view = runs.get_run_view(
                _project(settings, project), run_id.upper(), store_dir=settings.store_dir
            )
            m, r, s = view.meta, view.repro, view.stats
            snaps = view.snapshots[-limit:]
            return {
                "meta": _dump(m),
                "repro": {
                    "git_commit": r.git_commit,
                    "git_dirty": r.git_dirty,
                    "dataset_uri": r.dataset_uri,
                    "dataset_hash": r.dataset_hash,
                    "packages_lock_sha256": r.packages_lock_sha256,
                    "seed": r.seed,
                    "python_version": r.python_version,
                    "cuda": r.cuda,
                    "gpu_names": list(r.gpu_names),
                    "config_keys": len(r.config),
                },
                "metrics": _dump(s),
                "artifacts": [_dump(a) for a in view.artifacts[:limit]],
                "artifact_total": len(view.artifacts),
                "snapshots": [
                    {
                        "step": snap.step,
                        "eval_set_hash": snap.eval_set_hash,
                        "n_images": snap.n_images,
                        "task": snap.task,
                        "metrics": dict(snap.metrics),
                    }
                    for snap in snaps
                ],
                "snapshot_total": len(view.snapshots),
            }

    @server.tool(title="List baselines", annotations=read)
    def baselines_list(project: Project = None, history: bool = False) -> dict[str, Any]:
        """List the verified baselines (pinned evaluation snapshots) of a project."""
        from cvpilot.compare import baselines

        with _call("baselines_list") as settings:
            found = baselines.list_baselines(
                _project(settings, project), history=history, store_dir=settings.store_dir
            )
            return {"baselines": [_dump(b) for b in found[:MAX_LIMIT]], "total": len(found)}

    @server.tool(title="Show a baseline", annotations=read)
    def baseline_show(
        tag: Annotated[str, Field(description="Baseline tag (without 'tag:').")],
        project: Project = None,
    ) -> dict[str, Any]:
        """Show a baseline and its headline metrics, verifying the pin and its hash chain."""
        from cvpilot.compare import baselines

        with _call("baseline_show") as settings:
            view = baselines.show_baseline(
                _project(settings, project), tag, store_dir=settings.store_dir
            )
            return {
                "baseline": _dump(view.baseline),
                "metrics": dict(view.snapshot.metrics),
                "n_images": view.snapshot.n_images,
                "history": view.history,
                "head": {"seq": view.head_seq, "chain_sha256": view.head_chain_sha256},
            }

    @server.tool(title="Compare with a baseline", annotations=read)
    def compare(
        run_id: RunId,
        baseline: Baseline,
        project: Project = None,
        eval_set: EvalSet = None,
        step: Step = None,
        top_k: TopK = 10,
    ) -> dict[str, Any]:
        """Compare a run's evaluation with a baseline: headline deltas with intervals, the worst
        classes and slices, confusion change counts. Deltas are metric points (0..100).
        No verdict: use ``gate`` for pass/fail."""
        from cvpilot.compare import service

        with _call("compare") as settings:
            cmp = service.run_compare(
                _project(settings, project), run_id.upper(), baseline, eval_set=eval_set,
                step=step, store_dir=settings.store_dir,
            )  # fmt: skip
            return comparison_summary(cmp, top_k)

    @server.tool(title="Regression gate", annotations=read)
    def gate(
        run_id: RunId,
        baseline: Baseline,
        max_drop: Annotated[
            list[str], Field(description="NAME=POINTS per headline metric, e.g. mAP50_95=0.5.")
        ],
        per_class_max_drop: Annotated[
            list[str] | None, Field(description="NAME=POINTS for any single class.")
        ] = None,
        slice_max_drop: Annotated[
            list[str] | None, Field(description="KEY=VALUE:POINTS or KEY=*:POINTS.")
        ] = None,
        min_images: Annotated[int, Field(ge=1)] = 30,
        skip_small_slices: bool = False,
        max_skipped: Annotated[int, Field(ge=0)] = 0,
        gt: Annotated[
            str | None,
            Field(description="Trusted ground-truth Parquet: re-evaluate stored predictions."),
        ] = None,
        classes: Annotated[str | None, Field(description="Class names file, with gt.")] = None,
        project: Project = None,
        eval_set: EvalSet = None,
        step: Step = None,
        top_k: TopK = 10,
    ) -> dict[str, Any]:
        """Gate a run against a baseline; fail-closed: any violation, missing metric or
        unverifiable check fails. Give gt and classes to re-evaluate the stored predictions
        instead of trusting stored metrics. Returns the verdict, violations and the comparison."""
        from cvpilot.compare import service
        from cvpilot.compare.gate import Thresholds, parse_drop, parse_slice_drop
        from cvpilot.compare.report import gate_dict

        with _call("gate") as settings:
            if (gt is None) != (classes is None):
                raise Refused("gt and classes must be given together")

            def drops(
                values: list[str] | None, parse: Callable[[str], tuple[str, float]]
            ) -> dict[str, float]:
                out: dict[str, float] = {}
                for text in values or []:
                    name, limit = parse(text)
                    if name in out:
                        raise Refused(f"{name!r} is given twice")
                    out[name] = limit
                return out

            thresholds = Thresholds(
                max_drop=drops(max_drop, parse_drop),
                per_class_max_drop=drops(per_class_max_drop, parse_drop),
                slice_max_drop=drops(slice_max_drop, parse_slice_drop),
                min_images=min_images,
                skip_small_slices=skip_small_slices,
                max_skipped=max_skipped,
            )
            name = _project(settings, project)
            if gt is not None and classes is not None:
                # Re-evaluating loads the ground truth: run it as the command line does, in a child
                # process under the CLI's memory ceiling, never inside this long-lived server.
                return _gate_recompute(
                    settings, name, run_id.upper(), baseline, paths.existing(gt),
                    paths.existing(classes), max_drop, per_class_max_drop, slice_max_drop,
                    min_images, skip_small_slices, max_skipped, eval_set, step, top_k,
                )  # fmt: skip
            outcome = service.run_gate(
                name, run_id.upper(), baseline, thresholds,
                eval_set=eval_set, step=step, store_dir=settings.store_dir,
            )  # fmt: skip
            service.log_gate(outcome)
            verdict = gate_dict(outcome.result, thresholds)
            verdict["violations"] = verdict["violations"][:MAX_LIMIT]
            return {
                "passed": outcome.result.passed,
                "gate": verdict,
                "comparison": comparison_summary(outcome.comparison, top_k),
            }

    @server.tool(title="Verify checkpoints", annotations=read)
    def ckpt_verify(
        directory: Annotated[str, Field(description="Checkpoint directory (one per project).")],
        run: Annotated[str | None, Field(pattern=r"^[0-9A-Za-z]{26}$")] = None,
        kind: Literal["all", "last", "best"] = "all",
        require_trust: Literal["any", "anchored", "authenticated"] | None = None,
        project: Project = None,
        limit: Limit = 50,
    ) -> dict[str, Any]:
        """Verify checkpoints (size, structure, SHA-256, chain, trust policy) exactly as
        ``cvpilot ckpt verify``; ``ok`` is true only when every checkpoint verifies."""
        with _call("ckpt_verify") as settings:
            args = ["ckpt", "verify", str(paths.existing(directory, directory=True)), "--json"]
            args += ["--kind", kind]
            if run:
                args += ["--run", run.upper()]
            if require_trust:
                args += ["--require-trust", require_trust]
            name = project or settings.project
            if name:
                args += ["--project", _project(settings, name)]
            code, out, err = _cli(args, settings)
            if code == 2:
                raise Refused(scrub(safe_text(err.strip().splitlines()[-1] if err else "", 300)))
            rows = json.loads(out) if out.strip() else []
            return {
                "ok": code == 0,
                "checkpoints": rows[:limit],
                "total": len(rows),
                "notes": [ln for ln in err.splitlines() if ln.startswith(("note:", "error:"))][:20],
            }

    @server.tool(title="Check an alert config", annotations=read)
    def alerts_check(
        config: Annotated[str, Field(description="Alert configuration (JSON) file.")],
        run: Annotated[
            str | None,
            Field(pattern=r"^[0-9A-Za-z]{26}$", description="Replay this recorded run against it."),
        ] = None,
        project: Project = None,
    ) -> dict[str, Any]:
        """Validate an alert config, check that its channels can be built and, with ``run``,
        replay the run to show which alerts would fire. Nothing is sent."""
        with _call("alerts_check") as settings:
            args = ["alerts", "check", str(paths.existing(config))]
            if run:
                args += ["--run", run.upper(), "--project", _project(settings, project)]
            code, out, err = _cli(args, settings)
            lines = out.splitlines()
            return {
                "ok": code == 0,
                "lines": lines[:_LINES],
                "truncated": len(lines) > _LINES,
                "error": scrub(safe_text(err.strip().splitlines()[-1], 300))
                if code and err
                else None,
            }


def _register_write(server: MCPServer, create: ToolAnnotations) -> None:
    @server.tool(title="Pin a baseline", annotations=create)
    def baseline_set(
        tag: Annotated[str, Field(description="Baseline tag to pin.")],
        run_id: RunId,
        project: Project = None,
        eval_set: EvalSet = None,
        step: Step = None,
        replace: Annotated[
            bool, Field(description="Re-pin a tag that is already pinned; the history is kept.")
        ] = False,
    ) -> dict[str, Any]:
        """Pin a verified evaluation snapshot of a run as a baseline tag (gates then use
        'tag:<name>'). Nothing is deleted: re-pinning keeps the old pin in the history."""
        from cvpilot.compare import baselines

        with _call("baseline_set") as settings:
            pinned = baselines.set_baseline(
                _project(settings, project), tag, run_id.upper(), eval_set=eval_set, step=step,
                replace=replace, store_dir=settings.store_dir,
            )  # fmt: skip
            return {"baseline": _dump(pinned)}


INSTRUCTIONS = (
    "CV-Pilot: recorded training runs, evaluation snapshots, baselines and regression gates of "
    "computer-vision models. Start with runs_list or baselines_list, then run_show, compare or "
    "gate. Metric deltas are points on the 0..100 scale. Paths must lie under the configured "
    f"roots. baseline_set is present only when the server was started with {WRITE_ENV}=1."
)


def create_server(
    *, roots: tuple[Path, ...] | None = None, allow_write: bool | None = None
) -> MCPServer:
    """Build the server; ``None`` means: read ``CVPILOT_MCP_ROOTS`` or
    ``CVPILOT_MCP_ALLOW_WRITE``."""
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

    from cvpilot import __version__

    read = ToolAnnotations(
        read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
    )
    create = ToolAnnotations(  # pins a baseline; never deletes or rewrites history
        read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False
    )
    server = MCPServer(
        "cvpilot", version=__version__, instructions=INSTRUCTIONS, log_level="WARNING"
    )
    _register_read(server, Paths(roots if roots is not None else _env_roots()), read)
    if allow_write if allow_write is not None else _env_flag(WRITE_ENV):
        _register_write(server, create)
    return server


def main(run: Callable[[MCPServer], None] = lambda s: s.run("stdio")) -> None:
    """Serve over standard input and output until the client disconnects."""
    run(create_server())
