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

## User Guide

### Given opinion partitions for each of the topics, compute the consensus partition

```python
# import needed libraries
>>> import pandas as pd
>>> import multiway_alignment.consensus as mac

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
>>> mac.get_consensus_partition(opinions=df)
{
    "A0_B0_C1": {0},
    "A0_B1_C0": {1},
    "A1_B0_C1": {2},
    "A1_B1_C0": {3}
}

# this function is equivalent, but might be slower
>>> mac.get_consensus_partition_recursive(opinions=df)
{
    "A0_B0_C1": {0},
    "A0_B1_C0": {1},
    "A1_B0_C1": {2},
    "A1_B1_C0": {3}
}

# get list of labels for the consensus partition
>>> mac.get_consensus_labels(opinions=df)
['A0_B0_C1', 'A0_B1_C0', 'A1_B0_C1', 'A1_B1_C0']
```

### Given opinion partitions for each of the topics, compute the multiway alignment score of all of them

```python
# import needed libraries
>>> import pandas as pd
>>> import multiway_alignment.consensus as mac
>>> import multiway_alignment.score as mas

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

# get list of labels for the consensus partition
>>> partition_labels = mac.get_consensus_labels(opinions=df)

# compute 3-way alignment score using AMI (adjusted mutual info score)
# and adjust with the null model
>>> mas.multiway_alignment_score(
...     df, partition_labels, which_score="ami", adjusted=True,
... )
6.40685300762983e-16

# compute 3-way alignment score using NMI (normalized mutual info score)
# and adjust with the null model
>>> mas.multiway_alignment_score(
...     df, partition_labels, which_score="nmi", adjusted=True,
... )
0.0

# if we use NMI (normalized mutual info score) without adjusting it
# with a null model, the resulting score is inflated
>>> mas.multiway_alignment_score(
...     df, partition_labels, which_score="nmi", adjusted=False,
... )
0.6666666666666666
```
