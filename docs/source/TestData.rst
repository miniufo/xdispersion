# Test data

The NetCDF files under `data/` are scientific regression fixtures for ragged,
local, and non-local Lagrangian trajectories. They are intentionally used by
the reference suite and are not required for importing the package.

The fast CI suite excludes large regression and memory tests. Run the complete
reference suite locally with:

```bash
python -m pytest -m regression
```
