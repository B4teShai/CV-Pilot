# Dependencies and licensing

## Project license

CV-Pilot is Apache-2.0. Every source file is covered by `LICENSE`; attribution for
bundled third-party material lives in `NOTICE`.

## Core versus extras

- The core distribution installs on CPU-only machines in seconds and never pulls in
  PyTorch, CUDA, a web framework or a database server. Core runtime dependencies are listed
  in [ADR-0003](../adr/0003-repo-and-tooling.md) and checked by CI.
- Everything heavier is an optional extra (`pip install "cvpilot[name]"`). Extras are
  imported lazily, inside the function that needs them, and the import error names the
  extra to install.
- A feature that is only available with an extra says so in its `--help` text and in
  [../architecture/05-runtime.md](../architecture/05-runtime.md).

## License allow-list

| Core dependency | Optional extra (lazy import) | Separate plugin package | Never |
|---|---|---|---|
| Apache-2.0, MIT, BSD-2/3, ISC, PSF, MPL-2.0 | The core set plus copyleft libraries such as AGPL-3.0 / GPL-3.0 (for example Ultralytics, Cleanlab) | Anything that includes or derives from copyleft code (for example a YOLOE proposer) | Non-commercial or no-derivatives terms (for example CC BY-NC-ND) |

- Copyleft (AGPL/GPL) packages are never core dependencies and their code is never vendored.
  They may be declared as an optional extra and imported lazily inside the adapter that
  needs them. The adapter's `--help` text and the runtime docs state that using the extra
  makes the user's combined work subject to that package's license.
- Code that includes or derives from copyleft code ships as a separately named package
  that depends on this one, so this distribution stays Apache-2.0.
- Model weights under custom licenses (SAM License, DINOv3 License, gated Hugging Face
  repositories) are never vendored or redistributed. Users download them and accept the
  terms themselves; the registry shows the license on first use.
- Adding a dependency with any other license is an ADR.

## Adding or upgrading a dependency

1. Check the license against the table above and the transitive tree
   (`uv tree --package <name>`).
2. Add it to `pyproject.toml` in core or in the right extra; run `uv lock`.
3. Update [ADR-0003](../adr/0003-repo-and-tooling.md) if the core set changes.
4. CI's dependency check (`tools/qa_check.py` in the workspace, and the CI job) must still
   pass.

## Supply-chain hygiene

- Lockfile committed; CI installs with `uv sync --locked`.
- Direct dependencies carry a lower bound; upper bounds only when a known break exists.
- GitHub Actions are pinned to a major version at minimum; pinning to a SHA is preferred
  before the first public release.
