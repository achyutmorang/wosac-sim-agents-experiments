from __future__ import annotations

from src.platform import wosac_colab_runtime_config


def test_wosac_runtime_profile_defaults() -> None:
    cfg = wosac_colab_runtime_config(repo_url="https://github.com/achyutmorang/wosac-sim-agents-experiments.git")
    assert cfg.repo_branch == "main"
    assert cfg.repo_dir == "/content/wosac-sim-agents-experiments"
    assert cfg.required_drive_folder == "/content/drive/MyDrive/wosac_experiments"
    assert cfg.strict_lockfile_check is True
