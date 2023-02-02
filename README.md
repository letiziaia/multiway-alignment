# multilayer-alignment

This repository implements an algorithm for quantifying multilayer alignment.
You can refer to the [slide deck](https://docs.google.com/presentation/d/1HMEE5kOwwJPLBmAgycKIMSWRx0eCxd3RtSxVR1Jdczw/) for the general idea.

## Structure of the repo

- `\src\`: source code
- `\tests\`: tests for the source code

## Setting up

![python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)

You should have python 3.10 and [pipenv](https://github.com/pypa/pipenv#installation) installed.

```shell
# install dependencies (including dev)
$ pipenv install -d

# activate environment
$ pipenv shell
```

## Formatting

The project uses the [Black](https://black.readthedocs.io/en/stable/index.html) code formatter.
All the code can be formatted by running `python3 -m black .` in root dir.