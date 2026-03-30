from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_FORWARD_STUB = """    def forward(self, data) -> HeteroData:
        return self.__call__(data)

"""

_TORCH_LOAD_NEEDLE = "checkpoint = torch.load(filename, map_location=loc_type)"
_TORCH_LOAD_PATCH = "checkpoint = torch.load(filename, map_location=loc_type, weights_only=False)"


@dataclass(frozen=True)
class SmartCompatPatchResult:
    target_path: str
    applied: bool
    already_compatible: bool


def patch_waymo_target_builder(smart_repo_dir: str | Path) -> SmartCompatPatchResult:
    repo_dir = Path(smart_repo_dir).expanduser().resolve()
    target_builder_path = repo_dir / "smart" / "transforms" / "target_builder.py"
    if not target_builder_path.exists():
        raise FileNotFoundError(f"Missing SMART target builder: {target_builder_path}")

    original = target_builder_path.read_text(encoding="utf-8")
    if "def forward(self, data) -> HeteroData:" in original:
        return SmartCompatPatchResult(
            target_path=str(target_builder_path),
            applied=False,
            already_compatible=True,
        )

    needle = "    def __call__(self, data) -> HeteroData:\n"
    if needle not in original:
        raise RuntimeError(
            "Unable to locate WaymoTargetBuilder.__call__ for compatibility patching: "
            f"{target_builder_path}"
        )

    patched = original.replace(needle, _FORWARD_STUB + needle, 1)
    target_builder_path.write_text(patched, encoding="utf-8")
    return SmartCompatPatchResult(
        target_path=str(target_builder_path),
        applied=True,
        already_compatible=False,
    )


def patch_smart_checkpoint_loader(smart_repo_dir: str | Path) -> SmartCompatPatchResult:
    repo_dir = Path(smart_repo_dir).expanduser().resolve()
    smart_model_path = repo_dir / "smart" / "model" / "smart.py"
    if not smart_model_path.exists():
        raise FileNotFoundError(f"Missing SMART model file: {smart_model_path}")

    original = smart_model_path.read_text(encoding="utf-8")
    if _TORCH_LOAD_PATCH in original:
        return SmartCompatPatchResult(
            target_path=str(smart_model_path),
            applied=False,
            already_compatible=True,
        )
    if _TORCH_LOAD_NEEDLE not in original:
        raise RuntimeError(
            "Unable to locate SMART checkpoint loader for compatibility patching: "
            f"{smart_model_path}"
        )

    patched = original.replace(_TORCH_LOAD_NEEDLE, _TORCH_LOAD_PATCH, 1)
    smart_model_path.write_text(patched, encoding="utf-8")
    return SmartCompatPatchResult(
        target_path=str(smart_model_path),
        applied=True,
        already_compatible=False,
    )
