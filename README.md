# ADER-DG Transport

This repo contains code for the paper https://arxiv.org/abs/2507.07304. To install clone the repository and run:

```commandline
cd ader-dg-transport
python3 -m pip install -e .
```

To replicate the experiments in the paper run:
```commandline
cd experiments
bash run_convergence_experiments.sh
bash run_stability_1D.sh
python3 stability_proof_1D.py
```

For the 2D and 3D stability analysis edit the ncpus variable in `run_stability_2D.sh` and `run_stability_3D.sh`. We recommended using a large number CPUs for these tasks, and found 100 to be sufficient.