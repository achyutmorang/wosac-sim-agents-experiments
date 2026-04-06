from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_persist_processed_split_copies_validation_and_writes_manifest(tmp_path: Path) -> None:
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    validation = src_root / "validation"
    nested = validation / "nested"
    nested.mkdir(parents=True, exist_ok=True)
    (validation / "a.pkl").write_bytes(b"abc")
    (nested / "b.pkl").write_bytes(b"defg")

    manifest_path = tmp_path / "manifest.json"
    script = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "persist_processed_split.py"
    )

    subprocess.run(
        [
            "python3",
            str(script),
            "--src-root",
            str(src_root),
            "--dst-root",
            str(dst_root),
            "--split",
            "validation",
            "--manifest-json",
            str(manifest_path),
        ],
        check=True,
    )

    assert (dst_root / "validation" / "a.pkl").read_bytes() == b"abc"
    assert (dst_root / "validation" / "nested" / "b.pkl").read_bytes() == b"defg"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["split"] == "validation"
    assert manifest["destination"]["files"] == 2
    assert manifest["source"]["files"] == 2


def test_persist_processed_split_noops_when_source_is_already_durable(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    training = processed_root / "training"
    training.mkdir(parents=True, exist_ok=True)
    (training / "a.pkl").write_bytes(b"abc")

    manifest_path = tmp_path / "manifest.json"
    script = Path(__file__).resolve().parent.parent / "scripts" / "persist_processed_split.py"

    subprocess.run(
        [
            "python3",
            str(script),
            "--src-root",
            str(processed_root),
            "--dst-root",
            str(processed_root),
            "--split",
            "training",
            "--manifest-json",
            str(manifest_path),
        ],
        check=True,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["copy_mode"] == "already_durable"
    assert manifest["destination"]["files"] == 1
