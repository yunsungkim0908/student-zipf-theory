
# Student Zipf Theory

This repository contains the code used for the experiments in ["The Student Zipf Theory: Inferring Latent Structures in Open-Ended Student Work To Help Educators"](https://dl.acm.org/doi/abs/10.1145/3576050.3576116), published in LAK'23: 13th International Learning Analytics and Knowledge Conference. 

## Installation

Go to the root of the repository and run the following commands:
```
conda create --name zipf_theory python=3.9
conda activate zipf_theory
pip install -e .
cd zipf_inference
pip install -r requirements.txt
```
These commands will create a conda environment named `zipf_theory` and install the required packages. All source code is contained in the directory named `zipf_inference`.

## Zipf Inference

This repository has code for Zipf Inference (`exp/inference.py`), which can be used to conduct rank prediction and compute scores. This script takes the following inputs:
```
usage: inference.py [-h] --data DATA --out-name OUT_NAME [--device DEVICE] [--num-iter NUM_ITER]

options:
-h, --help show this help message and exit
--data DATA
--out-name OUT_NAME
--device DEVICE
--num-iter NUM_ITER
--scores-only
```
This script loads the data from the `[DATA_DIR]/[DATA]` directory and performs Zipf inference for `NUM_ITER` classrooms of 70 random students. Results are stored under `[OUT_DIR]/[OUT_NAME].pkl`, and the scores used for rank prediction are stored under `[OUT_DIR]/[OUT_NAME].scores.pkl`, which can be used for slope estimation as described in the paper. (When the `--scores-only` flag is set, only the scores file is saved.)

Each dataset must be stored in the `[DATA_DIR]/[DATA]` directory, which must contain the following files
 - `rank_cnts.json`: dictionary mapping rank to count
 - `rank_resps.json`: dictionary mapping rank to code
- `ranked_resps.json`: ordered list of code, ordered by count

## Ability Model Experiments

The code used for simulating the Fixed Ability Student Model (FASM) and Varied Ability Student Model (VASM) in Section 7 are in the `theory` directory. To replicate the simulation experiments, 
 1. Run `make vam_generate` to compile executables, and
 2. Run `bash generate_vam.sh`

The `generate_vam.sh` script calls the `vam_generate` executable, which takes the following arguments
```
Options:
--help  help message
--dist-only sample distributions only
--load-dist load distributions from label dir
--label arg label for binary output
--type arg  type of grammar
--num-choices arg (=5)  number of choices
--length arg (=10)  length of choices
--ability-bins arg (=0) number of discrete ability bins (default: unif[0,1].)
--uniformity arg (=-1)  uniformity param (default: unif[0,1])
--ability-lim arg (=0.2) upper and lower ability bounds (default: 0.2 -> [0.2,0.8].)
--num-fixed-ability arg (=2)  number of fixed ability values to evaluate (default: 2)
--num-traj arg (=1000)  number of trajectories sampled
--per-thread arg (=1000)  number of trajectories sampled per thread
```

Samples are stored under `theory/out/[OUT_NAME]`, where fixed ability model samples are stored in `[ability]_vals.bin`, and varied ability model samples are stored in `vals.bin`. The `get_sorted_counts(filename)` function under `theory/utils` can be used to load the sorted list of sample counts.
