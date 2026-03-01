#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize_slug(value: str) -> str:
    slug = ''.join(ch if (ch.isalnum() or ch in {'-', '_'}) else '-' for ch in str(value).strip().lower())
    slug = slug.strip('-_')
    if not slug:
        raise ValueError('Experiment slug cannot be empty.')
    return slug


def ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_if_missing(path: Path, content: str, overwrite: bool) -> str:
    if path.exists() and not overwrite:
        return 'skipped'
    ensure(path.parent)
    path.write_text(content)
    return 'created'


def scaffold(repo_root: Path, slug: str, title: str, objective: str, overwrite: bool) -> dict[str, list[str]]:
    slug = normalize_slug(slug)
    workflow_name = slug.replace('-', '_')

    paths = {
        'readme': repo_root / 'experiments' / slug / 'README.md',
        'config': repo_root / 'configs' / 'experiments' / f'{slug}.json',
        'workflow': repo_root / 'src' / 'experiments' / f'{workflow_name}.py',
    }

    readme = f"""# {title}

## Objective
{objective}

## Success Criteria
- At least one baseline run.
- At least one controlled variant.
- Metrics logged with run date and commit hash.
"""

    config = json.dumps(
        {
            'slug': slug,
            'title': title,
            'objective': objective,
            'run': {
                'run_tag_prefix': workflow_name,
                'persist_root': '/content/drive/MyDrive/wosac_experiments',
            },
        },
        indent=2,
    ) + '\n'

    workflow = f"""from __future__ import annotations


def run_{workflow_name}(**kwargs):
    return {{'status': 'todo', 'slug': '{slug}', 'kwargs': kwargs}}
"""

    created: list[str] = []
    skipped: list[str] = []
    for key, path in paths.items():
        content = {'readme': readme, 'config': config, 'workflow': workflow}[key]
        status = write_if_missing(path, content, overwrite)
        (created if status == 'created' else skipped).append(str(path))

    return {'created': created, 'skipped': skipped}


def main() -> int:
    parser = argparse.ArgumentParser(description='Scaffold a new WOSAC experiment pack.')
    parser.add_argument('--repo-root', default='.', help='Repository root path.')
    parser.add_argument('--slug', required=True, help='Experiment slug.')
    parser.add_argument('--title', required=True, help='Human-readable title.')
    parser.add_argument('--objective', required=True, help='One-line objective.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files.')
    args = parser.parse_args()

    summary = scaffold(
        repo_root=Path(args.repo_root),
        slug=args.slug,
        title=args.title,
        objective=args.objective,
        overwrite=bool(args.overwrite),
    )
    print('created:')
    for item in summary['created']:
        print(f'  - {item}')
    print('skipped:')
    for item in summary['skipped']:
        print(f'  - {item}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
