# Authorship Verification Using Distance-Based

## Data

***


## Prerequisites

***

- Python 3.6+ (we recommend the Anaconda Python distribution)
- scikit-learn, numpy, scipy
- non-essential: tqdm, seaborn/matplotlib
- pan20_verif_evaluator.py

## Run

***

- Using from the command line:
```sh
>> python pan20-verif-baseline.py \
          -input_pairs="/home/alexis/proyectSS/pan20-authorship-verification-training-dataset1/pairs.jsonl" \
          -input_truth="/home/alexis/proyectSS/Splits/small/truth.jsonl" \
          -test_pairs="/home/alexis/proyectSS/Splits/small/test/pairs.jsonl" \
          -num_iterations=0 \
          -output="out"
```