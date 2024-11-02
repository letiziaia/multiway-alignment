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

#### 1. Perfect Alignment

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
        "A": [0, 0, 0, 0, 1, 1, 1, 1],
        "B": [0, 0, 0, 0, 1, 1, 1, 1],
        "C": [1, 1, 1, 1, 0, 0, 0, 0],
    }
)


# compute 3-way alignment score using NMI (normalized mutual info score)
>>> mas.multiway_alignment_score(
...     df, which_score="nmi", adjusted=False,
... )
1.0

# compute 3-way alignment score using NMI (normalized mutual info score)
# and adjust with the null model
>>> mas.multiway_alignment_score(
...     df, which_score="nmi", adjusted=True,
... )
0.8767167706710732

# compute 3-way alignment score using AMI (adjusted mutual info score)
>>> mas.multiway_alignment_score(
...     df, which_score="ami", adjusted=False,
... )
1.0

# compute 3-way alignment score using AMI (adjusted mutual info score)
# and adjust with the null model
>>> mas.multiway_alignment_score(
...     df, which_score="ami", adjusted=True,
... )
0.933281539775369
```

In this example, we computed multiway alignment for a perfectly aligned system of 8 individuals and 3 topics.
Both the multiway alignment scores obtained by using NMI and by using AMI give 1 (perfect alignment). However, in case we are dealing with a sample of the population, we might want to account for the number of individuals, that here is quite small. To do so, we can adjust the scores with the null model. The resulting alignment score is still quite high, but accounts for the fact that this perfect alignment we are seeing among 8 individuals might be arising by chance.

With a growing number of individuals, the effect of alignment arising from random chance is smaller, and the score does not overfit:

```python
>>> n_individuals = 1000
>>> df = pd.DataFrame(
    {
        "A": [0] * int(n_individuals/2) + [1] * int(n_individuals/2),
        "B": [0] * int(n_individuals/2) + [1] * int(n_individuals/2),
        "C": [0] * int(n_individuals/2) + [1] * int(n_individuals/2),
    }
)


# compute 3-way alignment score using NMI (normalized mutual info score)
>>> mas.multiway_alignment_score(
...     df, which_score="nmi", adjusted=False,
... )
1.0

# compute 3-way alignment score using NMI (normalized mutual info score)
# and adjust with the null model
>>> mas.multiway_alignment_score(
...     df, which_score="nmi", adjusted=True,
... )
0.9991381194997214

# compute 3-way alignment score using AMI (adjusted mutual info score)
>>> mas.multiway_alignment_score(
...     df, which_score="ami", adjusted=False,
... )
1.0

# compute 3-way alignment score using AMI (adjusted mutual info score)
# and adjust with the null model
>>> mas.multiway_alignment_score(
...     df, which_score="ami", adjusted=True,
... )
0.9998316111697388
```

#### 2. No Alignment

```python
>>> n_individuals = 10000
>>> opinions = np.array([0] * int(n_individuals/2) + [1] * int(n_individuals/2))
>>> o1 = opinions.copy()
>>> np.random.shuffle(opinions)
>>> o2 = opinions.copy()
>>> np.random.shuffle(opinions)
>>> o3 = opinions.copy()
>>> df = pd.DataFrame(
    {
        "A": o1,
        "B": o2,
        "C": o3,
    }
)

# compute 3-way alignment score using NMI (normalized mutual info score)
>>> mas.multiway_alignment_score(
...     df, which_score="nmi", adjusted=False,
... )
0.0002596921754934203

# compute 3-way alignment score using NMI (normalized mutual info score)
# and adjust with the null model
>>> mas.multiway_alignment_score(
...     df, which_score="nmi", adjusted=True,
... )
0.00014352480953112452

# compute 3-way alignment score using AMI (adjusted mutual info score)
>>> mas.multiway_alignment_score(
...     df, which_score="ami", adjusted=False,
... )
0.00011540052022510332

# compute 3-way alignment score using AMI (adjusted mutual info score)
# and adjust with the null model
>>> mas.multiway_alignment_score(
...     df, which_score="ami", adjusted=True,
... )
5.4181170472338464e-05
```

For a random system, multiway alignment score approaches 0.

### Given opinion partitions, compute all the multiway alignment scores

```python
import multiway_alignment.score as mas

# the following line computes all multiway alignment scores
# and saves the python objects to joblib files
mas.maximal_alignment_curve(
        opinions=dataframe,
        which_score="ami",
        adjusted=False,
        dump_to=”resultfile”,
    )
```

To load the maximal alignment curve (considering only the maximum scores):

```python
r = load("resultfile_best")
max_scores = [(k, v[0], v[1]) for k, v in r.items()]
max_scores = pd.DataFrame(max_scores)
max_scores.columns = ["order", "score", "topics"]
```

To load the full results:

```python
r = load("resultfile_all")
points = [(k.split("+")[0], v, k.split("+")[1:]) for k, v in r.items()]
points = pd.DataFrame(points)
points.columns = ["order", "score", "topics"]
```
