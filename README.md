# Transformers for Supervised Tabular Learning

The official repository for the research done in the manuscript "Transformers for Supervised Tabular Learning" that is under review currently.

## Environment Setup

`requirements.txt` can be used to replicate the conda environment used in this research. (**Note:** May not work on all OS and all Graphic card setups)

## Overview

- `data`
  -   `preprocessing`
      - All preprocessing done on the the sourced datasets to clean, encode, and split.
- `all_experiments`
  - `outofbox_performance`
    - Scripts used to compare SAINT, TAB, FT, CAT, and XGBoost on supervised tabular tasks.
  - `attention_entropy`
    - Scripts used to determine the spread and concentration in attention between self and cross attention.
  - `fourier_embeddings`
    - Scripts used to compare different embedding schemes when used with tabular transformers.
- `model`
  - `testing_Model.py`
    - The first implementation of CAT and our version of FT used in research.
  - `saint.py` and `for_rtdl`
    - Both are helpers for using SAINT and FT.
- The rest is not important for experimentation and replication of results

## Credit

The `rdtl` package which contains the out-of-box FT-Transformer: https://github.com/yandex-research/rtdl 

The repository of SAINT: https://github.com/somepago/saint

The implementation of TabTransformer: https://github.com/lucidrains/tab-transformer-pytorch
