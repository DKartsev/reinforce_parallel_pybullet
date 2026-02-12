# Parallel REINFORCE on PyBullet (Colab)

This project contains a ready-to-run Google Colab notebook that implements a **parallel REINFORCE** agent with:

- learned value baseline
- GAE (Generalized Advantage Estimation)
- vectorized environments (`AsyncVectorEnv` / `SyncVectorEnv`)
- ablation experiments and efficiency analysis

## File

- `reinforce_parallel_pybullet_colab.ipynb` — main notebook (run end-to-end with **Run all**)

## What the notebook does

1. Installs dependencies and checks PyBullet env availability (with fallback env selection).
2. Sets reproducibility (seeds + version logging).
3. Implements policy/value networks in PyTorch.
4. Collects parallel rollouts from multiple environments.
5. Computes returns and advantages (GAE + done masking).
6. Trains and evaluates across multiple experiment configs.
7. Saves metrics, plots, and summaries to:
   - `/content/runs/<timestamp>/`

## Main outputs

- `learning_curve.png`
- `losses.png`
- `advantage_stats.png`
- `efficiency.png`
- `ablations_summary.png`
- `results_table.csv`
- `results_table.json`

## Quick start (Colab)

1. [Upload/open](https://colab.research.google.com/drive/1qDNPbwvV87QG9wkSOwAQKKmaPjgJ-pQ7?usp=sharing) `reinforce_parallel_pybullet_colab.ipynb` in Google Colab.
2. Optional: set runtime to GPU (CPU also works).
3. Run all cells.

## Notes

- This is an educational project focused on transparent implementation and analysis.
- Training quality on locomotion tasks can be highly seed-sensitive; use multiple seeds for robust conclusions.
