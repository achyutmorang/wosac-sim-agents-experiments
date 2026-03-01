# Experiment Packs

Each pack should answer one concrete question tied to WOSAC metrics, with Colab-resumable execution.

Required pack contents:
- One short objective.
- At least one Colab notebook in `experiments/<slug>/notebooks/`.
- One baseline + one variant at minimum.
- Exact config path(s).
- Run order and expected artifacts.
- Success/failure criteria before execution.

Scaffold a pack:

```bash
python3 scripts/new_experiment.py \
  --slug my-variant \
  --title "My Variant" \
  --objective "Test one controlled change against baseline"
```

Notebook standards:
- `notebooks/NOTEBOOK_DESIGN_CONTRACT.md`
- `notebooks/templates/paper_experiment_colab_template.ipynb`
