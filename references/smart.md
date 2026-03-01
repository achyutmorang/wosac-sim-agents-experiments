# SMART Reference Notes (for TrajTok-Oriented Study)

## Core Paper
- SMART paper (NeurIPS 2024): https://arxiv.org/abs/2405.19364
- Official codebase: https://github.com/rainmaker22/SMART

## Why SMART Matters Here
- SMART frames multi-agent simulation as token prediction with scalable training/inference patterns.
- TrajTok is commonly discussed as trajectory-tokenization-oriented; SMART provides concrete open code you can execute and inspect now.

## Suggested Reading-to-Code Loop
1. Read SMART sections on tokenization, architecture, and rollout procedure.
2. Map each section to concrete files in the SMART repo.
3. Re-implement one isolated component inside your `wosac-baseline` branch (not everything at once).
4. Evaluate only via official WOSAC metrics and keep deltas logged.

## Immediate Questions To Answer in This Repo
- Which SMART components transfer directly to your current compute budget?
- Which parts require simplification for stable, beginner-friendly experimentation?
- Which modifications improve `realism_meta_metric` without worsening safety rates?
