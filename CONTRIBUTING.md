# Contributing to xdispersion

Install development dependencies with `python -m pip install -e ".[test]"`.
Use `pytest -m "not slow and not regression and not memory"` for fast checks;
run `pytest -m regression` before releases. Build docs with pandoc installed.
Numerical changes should include a reproducible test or updated reference data.
