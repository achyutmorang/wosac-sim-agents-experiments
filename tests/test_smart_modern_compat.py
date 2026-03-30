from __future__ import annotations

from pathlib import Path

from src.platform.smart_modern_compat import patch_waymo_target_builder


def test_patch_waymo_target_builder_inserts_forward_stub(tmp_path: Path) -> None:
    target_builder = tmp_path / "SMART" / "smart" / "transforms" / "target_builder.py"
    target_builder.parent.mkdir(parents=True, exist_ok=True)
    target_builder.write_text(
        "from torch_geometric.data import HeteroData\n"
        "\n"
        "class WaymoTargetBuilder:\n"
        "    def __call__(self, data) -> HeteroData:\n"
        "        return data\n",
        encoding="utf-8",
    )

    result = patch_waymo_target_builder(tmp_path / "SMART")

    patched = target_builder.read_text(encoding="utf-8")
    assert result.applied is True
    assert result.already_compatible is False
    assert "def forward(self, data) -> HeteroData:" in patched
    assert patched.index("def forward(self, data) -> HeteroData:") < patched.index(
        "def __call__(self, data) -> HeteroData:"
    )


def test_patch_waymo_target_builder_is_idempotent(tmp_path: Path) -> None:
    target_builder = tmp_path / "SMART" / "smart" / "transforms" / "target_builder.py"
    target_builder.parent.mkdir(parents=True, exist_ok=True)
    target_builder.write_text(
        "from torch_geometric.data import HeteroData\n"
        "\n"
        "class WaymoTargetBuilder:\n"
        "    def forward(self, data) -> HeteroData:\n"
        "        return self.__call__(data)\n"
        "\n"
        "    def __call__(self, data) -> HeteroData:\n"
        "        return data\n",
        encoding="utf-8",
    )

    result = patch_waymo_target_builder(tmp_path / "SMART")

    assert result.applied is False
    assert result.already_compatible is True
