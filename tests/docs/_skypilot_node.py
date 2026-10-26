"""One "node" of the SkyPilot walk-through: runs examples/skypilot/train.py as SkyPilot would, with
the bucket served from a local directory under the ``s3`` scheme (so it outlives each process).

``argv``: bucket directory. Everything else comes from the environment, as on a real node.
"""

import runpy
import sys
import types
from pathlib import Path

import fsspec
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem

SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "skypilot" / "train.py"


def install_bucket(bucket: str) -> None:
    sys.modules.setdefault("s3fs", types.ModuleType("s3fs"))  # the backend check only imports it

    class Bucket(DirFileSystem):
        def __init__(self, **_options: object) -> None:
            super().__init__(path=bucket, fs=LocalFileSystem(auto_mkdir=True))

    fsspec.register_implementation("s3", Bucket, clobber=True)


if __name__ == "__main__":
    install_bucket(sys.argv[1])
    sys.argv = [str(SCRIPT)]
    runpy.run_path(str(SCRIPT), run_name="__main__")
