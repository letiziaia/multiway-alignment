# Run Book

## Overview

This repository contains code to extend pairwise measure of alignment
to N-wise case.


## Operational tasks

### Install all dependencies and activate the environment

```shell
# install environment
$ pipenv install --dev

# activate the environment
$ pipenv shell
```

### Run all tests

From root directory,

```shell
$ python -m unittest discover -v
````

### Given partitions for each of the layers, compute mutual clusters

```python
# import needed modules
>>> import pandas as pd
>>> from src.mutual_clusters import compute_mutual_clusters

# load the partitions labels to a pandas DataFrame
>>> df = pd.DataFrame(
    {
        # in layer A, nodes 0 and 1 have label 0,
        # nodes 2 and 3 have label 1
        "A": [0, 0, 1, 1],
        "B": [0, 1, 0, 1],
        "C": [1, 0, 1, 0]
    }
   )

# get mutual clusters
>>> compute_mutual_clusters(cluster_labels_df=df)
{
    "A0_B0_C1": {0},
    "A0_B1_C0": {1},
    "A1_B0_C1": {2},
    "A1_B1_C0": {3}
}
```