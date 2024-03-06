# Run Book

## Overview

This repository contains code to extend a pairwise measure of alignment
based on mutual information to an N-wise case.

## Development tasks

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
```

### Given opinion partitions for each of the topics, compute the consensus partition

```python
# import needed modules
>>> import pandas as pd
>>> from multilayer_alignment.consensus import get_consensus_partition

# load the opinion labels to a pandas DataFrame
>>> df = pd.DataFrame(
    {
        # on topic A, individuals 0 and 1 have opinion 0,
        # individuals 2 and 3 have opinion 1
        "A": [0, 0, 1, 1],
        "B": [0, 1, 0, 1],
        "C": [1, 0, 1, 0]
    }
)

# get consensus partition
>>> get_consensus_partition(opinions=df)
{
    "A0_B0_C1": {0},
    "A0_B1_C0": {1},
    "A1_B0_C1": {2},
    "A1_B1_C0": {3}
}
```

Alternatively:

```python
# import needed modules
>>> import pandas as pd
>>> from multilayer_alignment.consensus import get_consensus_partition_recursive

# load the partitions labels to a pandas DataFrame
>>> df = pd.DataFrame(
    {
        # on topic A, individuals 0 and 1 have opinion 0,
        # individuals 2 and 3 have opinion 1
        "A": [0, 0, 1, 1],
        "B": [0, 1, 0, 1],
        "C": [1, 0, 1, 0]
    }
)

# get consensus partition
>>> get_consensus_partition_recursive(opinions=df)
{
    "A0_B0_C1": {0},
    "A0_B1_C0": {1},
    "A1_B0_C1": {2},
    "A1_B1_C0": {3}
}
```
