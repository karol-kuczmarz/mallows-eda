# Mallows Model on Estimation of Distribution Algorithms

Estimation of Distribution Algorithm on Travelling Salesmen Problem with Mallows distribution

## Initialization

We decided to utilize `anaconda`. It offers fast project setup and quick modifications.

At first you need to create an environment.

```shell
conda env create -f conda.yml
```

Then, after enviroment activation (`conda activate mallows`) install the requirements.

```shell
pip install -r requirements.txt
```

At the end, we make our development code available in every location.

```shell
pip install --no-deps --no-build-isolation -e .
```

## Dataset

Download the TSP datasets with [script](./scripts/download_data.py) (`scripts/download_data.py`).
