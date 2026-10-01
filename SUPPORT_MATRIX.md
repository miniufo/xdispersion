# Supported functionality

| Area | Status | Notes |
| --- | --- | --- |
| Ragged Lagrangian trajectories | Supported | Uses `obs`, `ids`, and trajectory metadata |
| Local and non-local particle pairs | Supported | Covered by numerical reference tests |
| Relative dispersion and diffusivity | Supported | See `xdispersion.measures` |
| Structure functions and correlations | Supported | Includes longitudinal/transverse measures |
| FSLE and CIST | Supported | Requires valid separation/time inputs |
| PDF/CDF and anisotropy | Supported | Output dimensions follow xarray inputs |
| Analytic predictions | Supported | See `xdispersion.analytics` |
| Fokker–Planck integration | Experimental | Validate CFL conditions for each case |
| Plotting helpers | Supported | Matplotlib-based |
| Dask/chunked processing | Supported | Chunk sizes should be tuned to the dataset |
| `xrft` spectral extra | Optional | Install with `xdispersion[extra]` |
