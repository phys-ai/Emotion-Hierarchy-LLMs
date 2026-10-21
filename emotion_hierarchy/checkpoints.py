"""Preflight checkpoint files before handing them to model loaders."""

import json
import stat
from pathlib import Path, PureWindowsPath


def validate_checkpoint(directory: Path, *, storage_root: Path | None = None) -> None:
    directory = directory.resolve()
    storage_root = (storage_root or directory).resolve()

    def regular_file(path: Path) -> None:
        resolved = path.resolve()
        if not resolved.is_relative_to(storage_root):
            raise ValueError(f"Checkpoint file escapes its storage directory: {path}")
        if not stat.S_ISREG(path.stat().st_mode):
            raise ValueError(f"Checkpoint file must be a regular file: {path}")

    # Check indexes before opening them: opening a FIFO would block here too.
    for path in directory.rglob("*"):
        if path.name.endswith((".index.json", ".safetensors", ".bin")):
            regular_file(path)
            if not path.name.endswith(".index.json"):
                continue
            index = json.loads(path.read_text(encoding="utf-8"))
            weights = index.get("weight_map")
            if not isinstance(weights, dict) or not weights:
                raise ValueError(f"Invalid checkpoint weight_map: {path}")
            for name in weights.values():
                if not isinstance(name, str) or not name:
                    raise ValueError(f"Invalid checkpoint shard name: {name!r}")
                relative = Path(name)
                windows = PureWindowsPath(name)
                if relative.is_absolute() or windows.drive or windows.root or ".." in relative.parts or ".." in windows.parts:
                    raise ValueError(f"Unsafe checkpoint shard path: {name!r}")
                regular_file(path.parent / relative)


def prepare_checkpoint(model_name: str, revision: str | None) -> str:
    local = Path(model_name).expanduser()
    if local.is_dir():
        validate_checkpoint(local)
        return str(local.resolve())

    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=model_name, revision=revision))
    # Hub snapshots link to the same model cache's blobs directory.
    validate_checkpoint(snapshot, storage_root=snapshot.parent.parent)
    return str(snapshot)
