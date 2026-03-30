from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def test_smart_train_repro_adds_smart_repo_to_sys_path(tmp_path: Path, monkeypatch) -> None:
    smart_repo = tmp_path / "SMART"
    (smart_repo / "smart" / "utils").mkdir(parents=True, exist_ok=True)
    (smart_repo / "smart" / "__init__.py").write_text("", encoding="utf-8")
    (smart_repo / "smart" / "utils" / "__init__.py").write_text("", encoding="utf-8")
    (smart_repo / "smart" / "utils" / "config.py").write_text(
        "def load_config_act(path):\n"
        "    return {'config_path': path}\n",
        encoding="utf-8",
    )

    marker_path = tmp_path / "train_ok.txt"
    (smart_repo / "train.py").write_text(
        "from smart.utils.config import load_config_act\n"
        "cfg = load_config_act('dummy')\n"
        f"open({str(marker_path)!r}, 'w', encoding='utf-8').write(cfg['config_path'])\n",
        encoding="utf-8",
    )

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.random = types.SimpleNamespace(seed=lambda seed: None)

    fake_pl = types.ModuleType("pytorch_lightning")
    fake_pl.seed_everything = lambda seed, workers=True: None

    fake_torch = types.ModuleType("torch")
    fake_torch.manual_seed = lambda seed: None
    fake_torch.use_deterministic_algorithms = lambda *args, **kwargs: None
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda seed: None)
    fake_torch.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(deterministic=False, benchmark=False))

    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "pytorch_lightning", fake_pl)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "smart_train_repro.py"
    spec = importlib.util.spec_from_file_location("smart_train_repro_test", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smart_train_repro.py",
            "--smart-repo-dir",
            str(smart_repo),
            "--config",
            "configs/train/train_scalable.yaml",
            "--seed",
            "2",
        ],
    )

    assert module.main() == 0
    assert marker_path.exists()
    assert marker_path.read_text(encoding="utf-8") == "dummy"
