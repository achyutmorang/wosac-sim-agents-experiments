# Experiment Configs

Small JSON configs that keep Colab runtime parameters centralized and versionable.

## Conventions
- One file per experiment pack: `configs/experiments/<pack-slug>.json`.
- Keep orchestration/runtime knobs here.
- Put algorithmic logic in `src/` modules, not JSON.
- Track config changes with commit hash and run manifests.

## Mandatory Colab Runtime Fields
- `repo.url`, `repo.branch`, `repo.repo_dir`
- `run.run_name`, `run.run_prefix`, `run.persist_root`
- `run.n_shards`, `run.shard_id`
- `run.resume_from_existing`, `run.run_enabled`
