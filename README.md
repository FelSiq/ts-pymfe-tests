# Ts-pymfe tests
Tests for the pymfe expansion for handling univariate time-series data, comparing the efficiency of the time-series metafeatures against the standard supervised learning meta-features for time-series classification.

All methods for extracting time-series meta-features are available at https://github.com/FelSiq/ts-pymfe.

## Repository organization
Each experiment part is separated in a distinct, modularized, Jupyter notebook.

The sub-directories are enumerated in natural order, implying which order each notebook is supposed to be run.

In the "extra_results" directory, you can find multiple executions of the whole test using different random seeds, and statistical tests showing that the methods implemented in the ts-pymfe expansion are better for time-series data than the unsupervised methods already implemented in the pymfe package.

## Data
Get the CompEngine dataset in the following URL: https://www.comp-engine.org/
