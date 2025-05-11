Pipelines
---------

The analysis pipelines within this repository are contained in the ``analysis/`` directory and include the following:

**GALAH Preprocessing** – ``galah_filter``

- Applies quality, abundance, and orbital cuts to GALAH DR3 stars. Cross-matches with Gaia EDR3 for distances and orbital parameters, selecting metal-poor, high-eccentricity stars suitable for halo analysis.

**APOGEE Filtering** – ``apogee_filter``

- Filters APOGEE DR17 stars based on log g, SNR, abundance flags, and derived kinematics. Optionally queries Gaia DR3 for precise photogeometric distances to apply distance uncertainty cuts.

**Extreme Deconvolution** – ``XDPipeline``

- Performs uncertainty-aware Gaussian Mixture Model clustering through Extreme Deconvolution. Includes model fitting (``run_XD``), component selection (``compare_XD``), probabilistic assignment (``assigment_XD``), summary table generation (``table_results_XD``), and visualisation (``plot_XD``). Designed for reproducible functionality with the ability to save analysis and import previous results.

**Reduced Dimensionality GMM** – ``ReducedGMMPipeline``

- An additional clustering pipeline that mirrors the XD high-dimensional analysis but operates on a UMAP-reduced space. It applies Gaussian Mixture Models directly to the lower-dimensional projection, with the functionality to map these assignments back to the original stellar features.

**Dimensionality + Clustering Initial Visualisation** – ``investigate_umap``

- Explores and visualises how well stellar populations separate in low-dimensional UMAP space and tests the behaviour of unsupervised methods (GMM or HDBSCAN) before applying full clustering pipelines.