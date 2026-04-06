from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.platform.smart_preprocess_resumable import preprocess_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess SMART data shard-by-shard with durable progress markers and resume support."
    )
    parser.add_argument("--smart-repo-dir", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--progress-json", default="")
    parser.add_argument("--split", choices=("training", "validation", "testing"), required=True)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.set_defaults(skip_existing=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    state_dir = Path(args.state_dir).expanduser() if str(args.state_dir).strip() else None
    progress_json = Path(args.progress_json).expanduser() if str(args.progress_json).strip() else None

    print(
        json.dumps(
            {
                "smart_repo_dir": str(Path(args.smart_repo_dir).expanduser()),
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "state_dir": str(state_dir) if state_dir is not None else "",
                "progress_json": str(progress_json) if progress_json is not None else "",
                "split": args.split,
                "max_shards": int(args.max_shards),
                "log_every": int(args.log_every),
                "skip_existing": bool(args.skip_existing),
            },
            indent=2,
            sort_keys=True,
        )
    )

    result = preprocess_shards(
        smart_repo_dir=Path(args.smart_repo_dir).expanduser(),
        input_dir=input_dir,
        output_dir=output_dir,
        state_dir=state_dir,
        progress_json=progress_json,
        max_shards=int(args.max_shards),
        skip_existing=bool(args.skip_existing),
        log_every=int(args.log_every),
    )
    print("final_progress:")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
