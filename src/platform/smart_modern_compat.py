from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_FORWARD_STUB = """    def forward(self, data) -> HeteroData:
        return self.__call__(data)

"""


@dataclass(frozen=True)
class SmartCompatPatchResult:
    target_builder_path: str
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
            target_builder_path=str(target_builder_path),
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
        target_builder_path=str(target_builder_path),
        applied=True,
        already_compatible=False,
    )
