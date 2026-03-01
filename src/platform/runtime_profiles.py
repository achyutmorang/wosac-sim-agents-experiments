from __future__ import annotations

from .colab_runtime import ColabRuntimeConfig


def wosac_colab_runtime_config(
    repo_url: str,
    repo_branch: str = "main",
    repo_dir: str = "/content/wosac-sim-agents-experiments",
    required_drive_folder: str = "/content/drive/MyDrive/wosac_experiments",
) -> ColabRuntimeConfig:
    """Default Colab bootstrap profile for WOSAC runs."""
    return ColabRuntimeConfig(
        repo_url=str(repo_url),
        repo_dir=str(repo_dir),
        repo_branch=str(repo_branch),
        required_drive_folder=str(required_drive_folder),
        verify_drive_access_every_run=False,
        force_reinstall=False,
        auto_restart_after_setup=True,
        strict_lockfile_check=True,
        setup_cache_enabled=True,
        revalidate_core_imports_on_cache_hit=True,
        setup_cache_path="/content/.wosac_setup_cache.json",
        force_module_hot_reload=True,
    )
