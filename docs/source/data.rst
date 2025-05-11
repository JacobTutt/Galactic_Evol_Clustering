Data
----

This project combines astrometric and spectroscopic data to enable a detailed chemo-dynamical analysis of the Milky Way’s stellar populations. The datasets used include:

1. `Gaia Data Release 3 <https://www.cosmos.esa.int/web/gaia/dr3>`_

2. `APOGEE Data Release 17 <https://www.sdss4.org/dr17/irspec/dr_synopsis/>`_

3. `GALAH Data Release 3 <https://www.galah-survey.org/dr3/overview/>`_

The full data used for the project can be found here:  
`Research Project - Google Drive Data <https://drive.google.com/drive/u/1/folders/1kldDRZKpXAH3Szk839DRJjNsZq1ekQ2D>`_  
Although these are provided for completeness, they are not required for the basic functionality of the pipeline. The contents are outlined below:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Directory
     - Description
   * - ``data/raw/``
     - The full surveys datasets as well as Value Added Catalogues (VACs) before effective cuts have been applied. These are predominantly required in Notebook 0 which constructed the datasets used throughout the analysis. They are also used within the plotting of GMM results to provide background context on the overall data distribution (although this can be easily removed by removing the ``full_survey_file`` input path).
   * - ``data/filtered/``
     - Holds the resultant APOGEE and GALAH datasets after quality and scientific cuts have been applied (built in Notebook 0). These are also included directly in the GitHub repository.
   * - ``XD_Results/``
     - Stores intermediate outputs from all initialisation of the clustering pipelines, including Gaussian parameters (means and covariances), model selection scores (e.g., AIC/BIC), and assignment probabilities. This allows the user to recreate the results without rerunning the full pipeline (computationally expensive).

Additionally, the repository contains:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Directory
     - Description
   * - ``XD_Results/``
     - Stores intermediate outputs from all initialisations of the clustering pipelines, including Gaussian parameters (means and covariances), model selection scores (e.g., AIC/BIC), and assignment probabilities. This allows the user to recreate the results without rerunning the full pipeline (which is computationally expensive).

