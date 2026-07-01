"""alphatest — command-line interface for QuantLab.

Subcommands:
  run       — evaluate an expression and print core metrics
  shuffle   — shuffle leakage test on an expression
  list      — table of saved alphas from the SQLite DB
  verify    — re-run a saved alpha and check the signatures still match

The CLI runs entirely against the engine + analytics modules; no FastAPI
server needs to be running.  It uses the same parquet cache, universe
registry and GICS map as the API, so numbers are identical.
"""
