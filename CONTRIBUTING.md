# Contributing to xdispersion

Install the package and test dependencies in a clean Python environment:

```bash
python -m pip install -e ".[test]"
```

Run the fast test suite with:

```bash
python -m pytest -m "not slow and not regression and not memory"
```

Run the full numerical regression suite with:

```bash
python -m pytest -m regression
```

Build the documentation locally with:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
```

Before opening a pull request, run the tests, `ruff check xdispersion tests`,
and the documentation build. Changes to numerical measures should include a
small reproducible test or an updated reference calculation.
