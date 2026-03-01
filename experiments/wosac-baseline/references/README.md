# WOSAC Baseline References

This folder stores the literature corpus and download utilities used by `experiments/wosac-baseline/lit_survey.md`.

## Files
- `fetch_papers.sh`: reproducible downloader for benchmark/method PDFs.
- `pdfs/`: downloaded papers.
- `pdfs_manifest.txt`: file-size manifest snapshot.

## Usage
```bash
cd experiments/wosac-baseline/references
./fetch_papers.sh
```

## Access Note
Some direct technical-report storage links for WOSAC 2025 winners currently return `403` for public requests. The script treats these as optional and continues.
