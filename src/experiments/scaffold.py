from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .layout import experiment_pack_paths
from .spec import normalize_slug


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, content: str, *, overwrite: bool) -> bool:
    if path.exists() and (not overwrite):
        return False
    _ensure_parent(path)
    path.write_text(content)
    return True


def _default_notebook(title: str, slug: str) -> Dict[str, Any]:
    return {
        'cells': [
            {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [
                    f'# {title}\n',
                    '\n',
                    '## Objective\n',
                    '- Run one WOSAC-focused experiment variant with Colab-resumable execution.\n',
                    '- Keep notebook orchestration-only and move logic to `src/` modules.\n',
                    '- Persist all outputs/checkpoints to Drive-backed storage.\n',
                ],
            },
            {
                'cell_type': 'code',
                'metadata': {},
                'execution_count': None,
                'outputs': [],
                'source': [
                    '# Step 1: Repo sync + runtime bootstrap (idempotent)\n',
                    'import os\n',
                    'import subprocess\n',
                    'import sys\n',
                    '\n',
                    "REPO_URL = 'https://github.com/achyutmorang/wosac-sim-agents-experiments.git'\n",
                    "REPO_DIR = '/content/wosac-sim-agents-experiments'\n",
                    '\n',
                    'if os.path.isdir(REPO_DIR):\n',
                    "    subprocess.run(['git', '-C', REPO_DIR, 'fetch', 'origin'], check=True)\n",
                    "    subprocess.run(['git', '-C', REPO_DIR, 'checkout', 'main'], check=True)\n",
                    "    subprocess.run(['git', '-C', REPO_DIR, 'pull', '--ff-only', 'origin', 'main'], check=True)\n",
                    'else:\n',
                    "    subprocess.run(['git', 'clone', '--depth', '1', '-b', 'main', REPO_URL, REPO_DIR], check=True)\n",
                    '\n',
                    'os.chdir(REPO_DIR)\n',
                    'for p in (REPO_DIR, os.path.join(REPO_DIR, \"src\")):\n',
                    '    if p not in sys.path:\n',
                    '        sys.path.insert(0, p)\n',
                    '\n',
                    'from src.platform import bootstrap_colab_runtime_with_config, wosac_colab_runtime_config\n',
                    'runtime_cfg = wosac_colab_runtime_config(repo_url=REPO_URL, repo_branch=\"main\", repo_dir=REPO_DIR)\n',
                    'bootstrap = bootstrap_colab_runtime_with_config(runtime_cfg)\n',
                    'print(\"repo_rev:\", bootstrap.repo_sync.repo_rev)\n',
                    '\n',
                    'if bootstrap.setup.result.get(\"restart_required\", False):\n',
                    "    raise RuntimeError('Runtime restart required. Rerun this cell after restart.')\n",
                ],
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    '# Step 2: Load config + pack metadata\n',
                    "from src.workflows import bootstrap_experiment_pack, load_experiment_config\n",
                    f"EXPERIMENT_SLUG = '{slug}'\n",
                    "bundle = bootstrap_experiment_pack(slug=EXPERIMENT_SLUG, repo_root='.')\n",
                    "cfg = load_experiment_config(slug=EXPERIMENT_SLUG, repo_root='.')\n",
                    "print('Experiment:', EXPERIMENT_SLUG)\n",
                    "print('Config run block:', cfg.get('run', {}))\n",
                    "display(bundle.summary_table)\n",
                ],
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    '# Step 3: Fast-fail checks before full run\n',
                    "RUN = cfg.get('run', {})\n",
                    "assert RUN.get('persist_root'), 'persist_root is required'\n",
                    "assert int(RUN.get('n_shards', 1)) >= 1\n",
                    "print('Fast-fail checks passed.')\n",
                ],
            },
        ],
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


def scaffold_experiment_pack(
    *,
    repo_root: str | Path,
    slug: str,
    title: str,
    objective: str,
    overwrite: bool = False,
) -> Dict[str, List[str]]:
    pack_slug = normalize_slug(slug)
    paths = experiment_pack_paths(repo_root=repo_root, slug=pack_slug)

    created: List[str] = []
    skipped: List[str] = []

    pack_readme = f"""# {title}

## Objective
{objective}

## Contents
- Notebook: `experiments/{pack_slug}/notebooks/{pack_slug}_colab.ipynb`
- Workflow: `src/workflows/{pack_slug.replace('-', '_')}_flow.py`
- Config: `configs/experiments/{pack_slug}.json`
- Paper-specific module: `src/experiments/papers/{pack_slug.replace('-', '_')}/`
"""
    if _write_text(paths['pack_dir'] / 'README.md', pack_readme, overwrite=overwrite):
        created.append(str(paths['pack_dir'] / 'README.md'))
    else:
        skipped.append(str(paths['pack_dir'] / 'README.md'))

    config_payload = {
        'slug': pack_slug,
        'title': title,
        'objective': objective,
        'repo': {
            'url': 'https://github.com/achyutmorang/wosac-sim-agents-experiments.git',
            'branch': 'main',
            'repo_dir': '/content/wosac-sim-agents-experiments',
        },
        'run': {
            'run_name': 'dev',
            'run_prefix': pack_slug.replace('-', '_'),
            'persist_root': '/content/drive/MyDrive/wosac_experiments',
            'n_shards': 1,
            'shard_id': 0,
            'resume_from_existing': True,
            'run_enabled': True,
        },
    }
    cfg_text = json.dumps(config_payload, indent=2, sort_keys=True) + '\n'
    if _write_text(paths['config_file'], cfg_text, overwrite=overwrite):
        created.append(str(paths['config_file']))
    else:
        skipped.append(str(paths['config_file']))

    module_init = (
        '"""Paper-specific reusable code for this experiment pack."""\n\n'
        '__all__ = []\n'
    )
    if _write_text(paths['module_dir'] / '__init__.py', module_init, overwrite=overwrite):
        created.append(str(paths['module_dir'] / '__init__.py'))
    else:
        skipped.append(str(paths['module_dir'] / '__init__.py'))

    workflow_stub = f"""from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class {pack_slug.replace('-', ' ').title().replace(' ', '')}Bundle:
    summary: Dict[str, Any]


def run_{pack_slug.replace('-', '_')}_flow(**kwargs: Any) -> {pack_slug.replace('-', ' ').title().replace(' ', '')}Bundle:
    return {pack_slug.replace('-', ' ').title().replace(' ', '')}Bundle(summary={{'status': 'todo', 'kwargs': dict(kwargs)}})
"""
    if _write_text(paths['workflow_file'], workflow_stub, overwrite=overwrite):
        created.append(str(paths['workflow_file']))
    else:
        skipped.append(str(paths['workflow_file']))

    notebook_obj = _default_notebook(title=title, slug=pack_slug)
    notebook_text = json.dumps(notebook_obj, indent=1) + '\n'
    if _write_text(paths['notebook_file'], notebook_text, overwrite=overwrite):
        created.append(str(paths['notebook_file']))
    else:
        skipped.append(str(paths['notebook_file']))

    return {'created': created, 'skipped': skipped}
