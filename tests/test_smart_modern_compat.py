from __future__ import annotations

from pathlib import Path

from src.platform.smart_modern_compat import patch_smart_checkpoint_loader, patch_waymo_target_builder


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


def test_patch_smart_checkpoint_loader_sets_weights_only_false(tmp_path: Path) -> None:
    smart_model = tmp_path / "SMART" / "smart" / "model" / "smart.py"
    smart_model.parent.mkdir(parents=True, exist_ok=True)
    smart_model.write_text(
        "def load_params_from_file(self, filename, logger, to_cpu=False):\n"
        "    loc_type = None\n"
        "    checkpoint = torch.load(filename, map_location=loc_type)\n",
        encoding="utf-8",
    )

    result = patch_smart_checkpoint_loader(tmp_path / "SMART")

    patched = smart_model.read_text(encoding="utf-8")
    assert result.applied is True
    assert result.already_compatible is False
    assert "torch.load(filename, map_location=loc_type, weights_only=False)" in patched


def test_patch_smart_checkpoint_loader_is_idempotent(tmp_path: Path) -> None:
    smart_model = tmp_path / "SMART" / "smart" / "model" / "smart.py"
    smart_model.parent.mkdir(parents=True, exist_ok=True)
    smart_model.write_text(
        "def load_params_from_file(self, filename, logger, to_cpu=False):\n"
        "    loc_type = None\n"
        "    checkpoint = torch.load(filename, map_location=loc_type, weights_only=False)\n",
        encoding="utf-8",
    )

    result = patch_smart_checkpoint_loader(tmp_path / "SMART")

    assert result.applied is False
    assert result.already_compatible is True
