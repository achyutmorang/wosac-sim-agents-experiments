# WOSAC Sim Agents 2025 Leaderboard References

Status verified on **March 1, 2026**.

## Top Methods and Public Artifacts
| Place | Method | Technical Report | Public Code Status |
|---|---|---|---|
| First | TrajTok | https://storage.googleapis.com/waymo-uploads/files/research/challenges/2025/technical_reports/sim_agents/trajtok.pdf | No official public code repository linked in report/challenge page as of Mar 1, 2026. |
| Second | RLFTSim | https://storage.googleapis.com/waymo-uploads/files/research/challenges/2025/technical_reports/sim_agents/rlftsim.pdf | Project page exists, but no full public training/inference repo identified from official links. |
| Third | comBOT | https://storage.googleapis.com/waymo-uploads/files/research/challenges/2025/technical_reports/sim_agents/combot.pdf | No official public code repository linked in report/challenge page as of Mar 1, 2026. |
| Honorable Mention | UniMM | https://storage.googleapis.com/waymo-uploads/files/research/challenges/2025/technical_reports/sim_agents/unimm.pdf | Public repo exists (`Longzhong-Lin/UniMM`), but README still says full code release is coming soon. |

## Official Benchmark and Metrics Sources
- Challenge page: https://waymo.com/open/challenges/2025/sim-agents/
- Official simulation tutorial notebook:
  - https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_sim_agents.ipynb
- Official metrics implementation:
  - https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/metrics.py
- Official feature implementations:
  - https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/trajectory_features.py
  - https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/interaction_features.py
  - https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/map_metric_features.py
- Official 2025 metric weights/config:
  - https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/challenge_2025_sim_agents_config.textproto

## Practical Code You Can Study Now
These are not necessarily top-2025 winning codebases, but they are public and relevant for strong baselines:

- SMART (NeurIPS 2024): https://github.com/rainmaker22/SMART
- TrafficBotsV1.5 (WOSAC 2024 3rd place): https://github.com/zhejz/TrafficBotsV1.5
- UniMM project repo (paper/resources status): https://github.com/Longzhong-Lin/UniMM

## Notes
- Keep this file strict: only verified links.
- If code is unavailable, track report + reproducibility notes instead of guessing implementation details.
