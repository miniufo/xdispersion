Supported functionality
========================

.. list-table::
   :header-rows: 1

   * - Area
     - Status
     - Notes
   * - Ragged Lagrangian trajectories
     - Supported
     - Uses ``obs``, ``ids``, and trajectory metadata.
   * - Local and non-local particle pairs
     - Supported
     - Covered by numerical reference tests.
   * - Relative dispersion and diffusivity
     - Supported
     - See ``xdispersion.measures``.
   * - FSLE and CIST
     - Supported
     - Requires valid separation and time inputs.
   * - Fokker–Planck integration
     - Experimental
     - Validate CFL conditions for each case.
   * - Dask/chunked processing
     - Supported
     - Chunk sizes should be tuned to the dataset.
   * - ``xrft`` spectral extra
     - Optional
     - Install with ``xdispersion[extra]``.
