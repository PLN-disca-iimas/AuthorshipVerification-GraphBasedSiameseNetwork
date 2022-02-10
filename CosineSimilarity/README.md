# Authorship Verification Distance-Based Baseline

## Data

***

The data folder contains datasets for train and test in .jsonl format

## Prerequisites

***

- Python 3.6+ (we recommend the Anaconda Python distribution)
- scikit-learn, numpy, scipy
- non-essential: tqdm, seaborn/matplotlib
- pan20_verif_baseline.py

## Run

***

- Using from the command line:
```sh
>> python pan20-verif-baseline.py \
          -input_pairs="/homePath/pan20-authorship-verification-training-dataset1/pairs.jsonl" \
          -input_truth="/homePath/truth.jsonl" \
          -test_pairs="/homePath/pairs.jsonl" \
          -num_iterations=0 \
          -output="out"
```