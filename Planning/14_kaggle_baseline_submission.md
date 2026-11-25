# 14 · Submit the safe pid_water notebook to Kaggle

**Phase:** Submission · **Priority:** P0 · **Size:** S · **Depends on:** nothing

## Goal

Prove the Kaggle interface works with a submission that cannot do worse than the kit's best baseline. This can be done
today and does not wait for discovery.

## Steps

1. `kaggle/submission.ipynb` subclasses `smart_buildings_inference_server.Policy` through `code/submission/`
   (`TransferPolicyCore`); without an attached `best.py` it runs pid_water.
2. Build the dataset: `python code/package_submission.py --out dist/kaggle_dataset` (26 files; it checks the package
   imports without the kit). Upload it as a Kaggle Dataset and attach it to the notebook.
3. Run the notebook on Kaggle against the public example scenario; then submit.

## Done when

- A public leaderboard score is recorded for the pid_water notebook (expected near the kit's pid_water level), and the
  notebook version that produced it is committed.
