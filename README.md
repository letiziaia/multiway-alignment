# multilayer-alignment

This repository implements an algorithm for quantifying multilayer alignment.
You can refer to the [slide deck](https://docs.google.com/presentation/d/1HMEE5kOwwJPLBmAgycKIMSWRx0eCxd3RtSxVR1Jdczw/) for the general idea.

## Structure of the repo

- `\multilayer_alignment\`: source code
- `\tests\`: tests for the source code

## Setting up

![python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)

You should have python >= 3.8 and [pipenv](https://github.com/pypa/pipenv#installation) installed.

```shell
# install dependencies (including dev)
$ pipenv install --dev

# activate environment
$ pipenv shell
```

## Formatting

The project uses [Black](https://black.readthedocs.io/en/stable/index.html), [flake8](https://flake8.pycqa.org/en/latest/) and [ruff](https://docs.astral.sh/ruff/) code linting.
All the code can be formatted by running `python3 -m black .` in root dir.
Additional issues can be found by running `python3 -m flake8 .` and `python3 -m ruff check .` in root dir.

## Tests

This code has test coverage for python 3.8, 3.9, 3.10, and 3.11. Current test coverage is at 89%.
