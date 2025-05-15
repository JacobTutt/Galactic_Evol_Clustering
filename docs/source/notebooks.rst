Notebooks
---------

The notebooks in this repository serve as descriptions of the pipeline design, motivation, and implementation, while also presenting key results and providing brief commentary on the research decisions made.

.. list-table::
   :widths: 35 80
   :header-rows: 1

   * - Notebook
     - Description
   * - ``0_data_preprocessing.ipynb``
     - Overview of the preprocessing and filtering pipelines, including data querying and the motivation behind the cuts used to obtain a usable dataset for analysis.
   * - ``1_apogee_scaled_pipeline.ipynb``
     - A full run-through of the Extreme Deconvolution (GMM) pipeline applied to the 6-dimensional APOGEE dataset (with normalisation), describing the analysis and providing visualisation and discussion of results.
   * - ``2_galah_scaled_pipeline.ipynb``
     - A similar pipeline applied to the 12-dimensional GALAH data.
   * - ``3_apogee_unscaled_pipeline.ipynb``
     - A demonstration of the XD pipeline’s handling of manually adjusted or unscaled energy dimensions.
   * - ``4_dimreduction_investigate.ipynb``
     - Uses dimensionality reduction to explore structure stability and investigate clustering performance in low-dimensional space.
   * - ``5_apogee_assignment_statistic.ipynb``
     - A Notebook demonstrating the cluster assignment analysis pipeline and investigating the Aurora Population.
   * - ``[6_galah_reconstruction.ipynb``
     - Overview of the pipeline that automates robust dimensionality mapping, clustering, and analysis on GALAH data, along with visualised results.
   * - ``7_6D_galah_reconstruction.ipynb``
     - Here we investigate the success of GALAH at resolving these structures and whether this is simply attributed to the Higher Dimensionality of Chemical Abundances or the dataset itself.