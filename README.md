# **Unveiling the Milky Way’s Formation History: Resolving Chemo-Dynamical Substructures in APOGEE and GALAH**

## Author: Jacob Tutt, Department of Physics, University of Cambridge
### Supervisor: Dr GyuChul Myeong, Institute of Astronomy, University of Cambridge

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)


## Description

The primary objective of this project is to investigate the reproducability of results presented in the paper: [Milky Way’s Eccentric Constitutents with Gaia, APOGEE, and
GALAH, Myeong et al 2022](https://iopscience.iop.org/article/10.3847/1538-4357/ac8d68). In addition to replicating the original analysis, this work explores the use of dimensionality reduction techniques to:

1. Better understand the coherence of stellar populations identified through high dimensional clustering.

2. Compare the success of the APOGEE and GALAH surveys in recovering the Milky Way’s assembly history.

3. Prepose suggestions to future approaches of clustering algorthims to improve convergence, scalability and stability for the application to the larger scale datasets from upcoming spectroscopic surveys such as WEAVE and 4MOST. 

This repository forms part of the submission for the MPhil in Data Intensive Science's Research Project at the University of Cambridge.


## Table of Contents
- [Data](#data)
- [Pipeline Functionalities](#pipelines)
- [Documentation](#documentation)
- [Notebooks](#notebooks)
- [Installation](#installation-and-usage)
- [License](#license)
- [Support](#support)
- [Author](#author)


## Data

This project combines astrometric and spectroscopic data to enable a detailed chemo-dynamical analysis of the Milky Way’s stellar populations. The datasets used include:

1. [Gaia Data Release 3](https://www.cosmos.esa.int/web/gaia/dr3)

2. [APOGEE Data Release 17](https://www.sdss4.org/dr17/irspec/dr_synopsis/)

3. [GALAH Data Release 3](https://www.galah-survey.org/dr3/overview/)

The full data used for the project can be found here:
[Research Project - Google Drive Data](https://drive.google.com/drive/u/1/folders/1kldDRZKpXAH3Szk839DRJjNsZq1ekQ2D). Although these are provided for completeness they are not required for the basic functionality of the pipeline. The contents is outlined below:

| Directory      | Description |
|----------------|-------------|
| `data/raw/`         | The full surveys datasets as well as Value Added Catalogues (VACs) before effective cuts have been applied. These are only required for [Notebook 0](0_data_preprocessing.ipynb) which constructes the datasets used throughout the analysis. Some files are also used within the ploting of GMM results to provide background context on the overall data distribution (however this functionality has been turned off so it is not required). |
| [`data/filtered/`](data/filtered/)    | Holds the resultant APOGEE and GALAH datasets after quality and scientific cuts have been applied (built in [Notebook 0](0_data_preprocessing.ipynb)). These are also included directly in the GitHub repository. |

Additionally the repository contains:

| Directory      | Description |
|----------------|-------------|
| [`XD_Results/`](XD_results/) |  Stores intermediate outputs from all initialisation of the clustering pipelines, including Gaussian parameters (means and covariances), model selection scores (e.g., AIC/BIC), and assignment probabilities. This allows the user to recreate the results without rerunning the full pipeline (computationally expensive). |

## Pipelines
The analysis pipelines within this repository are constained with the [analysis/](analysis/) directory and contains the following: 

**GALAH Preprocessing** – ([galah_filter](analysis/PreProcess.py)):

- Applies quality, abundance, and orbital cuts to GALAH DR3 stars. Cross-matches with Gaia EDR3 for distances and orbital parameters, selecting metal-poor, high-eccentricity stars suitable for halo analysis.

**APOGEE Filtering** – ([apogee_filter](analysis/PreProcess.py)):

- Filters APOGEE DR17 stars based on log g, SNR, abundance flags, and derived kinematics. Optionally queries Gaia DR3 for precise photogeometric distances to apply distance uncertainty cuts.

**Extreme Deconvolution** - ([XDPipeline](analysis/XD.py)):

- Performs uncertainty-aware Gaussian Mixture Model clustering through Extreme Deconvolution. Includes model fitting (run_XD), component selection (compare_XD), probabilistic assignment (assigment_XD), summary table generation (table_results_XD), and visualisation (plot_XD). Designed for reproducable functionality with ability to save analysis and import previous results. 

**Reduced Dimensionality GMM** - ([ReducedGMMPipeline](analysis/Reduced_GMM.py)):

- An additional clustering pipeline that mirrors the XD high dimensional analysis but operates on a UMAP-reduced space. It applies Gaussian Mixture Models directly to the lower-dimensional projection, with the functionality to exploit these assignments to map these back to disitributions of the original features of the stellar data.

**Dimensionality + Clustering Inital Visualisation** - ([investigate_umap](analysis/Dimreduce.py)):

- Explores and visualises how well stellar populations separate in low-dimensional UMAP space as well as test the behaviour of unsupervised methods (GMM or HDBSCAN) before applying full clustering pipelines.

## Documentation
Detailed documentation of the available pipelines is available [here](https://galactic-evol-clustering.readthedocs.io/en/latest/).

## Notebooks

The [notebooks](notebooks) in this repository serve as descriptions of the pipeline design, motivation, and implementation, while also presenting key results and providing brief commentary on the research decisions made.

| Notebook | Description |
|----------|-------------|
| [0_data_preprocessing.ipynb](notebooks/0_data_preprocessing.ipynb) | Overview of the preprocessing and filtering pipelines, including data querying and the motivation behind the cuts used to obtain a usable dataset for analysis. |
| [1_apogee_scaled_pipeline.ipynb](notebooks/1_apogee_scaled_pipeline.ipynb) | A full run-through of the Extreme Deconvolution (GMM) pipeline applied to the 6-dimensional APOGEE dataset (with normalisation), describing the analysis and providing visualisation and discussion of results. |
| [2_galah_scaled_pipeline.ipynb](notebooks/2_galah_scaled_pipeline.ipynb) | A similar notebook to the one above, but applied to the 12-dimensional space provided by GALAH. |
| [3_apogee_unscaled_pipeline.ipynb](notebooks/3_apogee_unscaled_pipeline.ipynb) | A demonstration of the XD pipeline’s functionality for automating the scaling procedure or manually adjusting energy (although results are not promising and not discussed in detail). |
| [4_dimreduction_investigate.ipynb](notebooks/4_dimreduction_investigate.ipynb) | A demonstration of dimensionality reduction techniques used to visualise the structures stability and an initial investigation into clustering within this space. |
| [5_apogee_assignment_statistic.ipynb](notebooks/5_apogee_assignment_statistic.ipynb) | A Notebook demonstrating the cluster assignment analysis pipeline and investigating the Aurora Population |
| [6_galah_reconstruction.ipynb](notebooks/6_galah_reconstruction.ipynb) | Overview of the pipeline that automates robust dimensionality mapping, clustering, and analysis on GALAH data, along with visualised results. |
| [7_6D_galah_reconstruction.ipynb](notebooks/7_6D_galah_reconstruction.ipynb) | Here we investigate the success of GALAH at resolving these structures and whether this is simply attributed to the Higher Dimensionality of Chemical Abundances or the dataset itself. |
| [8_computational_cost_compare.ipynb](notebooks/8_computational_cost_compare.ipynb) | Here we explicity compare the computational costs of the low dimensional and high dimensional pipelines to show the significant speed up across all models. |

---

## Installation and Usage

To reproduce the analysis in these notebooks, please follow these steps:

### 1. Clone the Repository

Clone the repository from the remote repository (GitLab) to your local machine.

```bash
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/jlt67.git
cd jlt67
```

### 2. Create a Fresh Virtual Environment
It is recommended to use a clean virtual environment to avoid dependency conflicts.
```bash
python -m venv env
source env/bin/activate   # For macOS/Linux
env\Scripts\activate      # For Windows
```

### 3. Install the dependencies
#### 3.1 Navigate to the repository’s root directory and install the PyPi package dependencies:
```bash
cd jlt67
pip install -r requirements.txt
```
#### 3.2 Additional Installation: Extreme Deconvolution

This project utilises the `extreme_deconvolution` algorithm for Gaussian Mixture Model (GMM) implementation. This is provided by Bovy et al. (2011) but is not directly available via PyPI.


To install the package, follow these steps, or for more details see the source repository's `README`:

**Source Repository:**

* GitHub: [https://github.com/jobovy/extreme-deconvolution](https://github.com/jobovy/extreme-deconvolution)

**Installation Steps:**

1.  **Clone the Repository**

    ```bash
    git clone [https://github.com/jobovy/extreme-deconvolution.git](https://github.com/jobovy/extreme-deconvolution.git)
    cd extreme-deconvolution
    ```

2.  **Build the Library**

    ```bash
    make
    ```

3.  **Install the Python Wrapper**

    ```bash
    make pywrapper
    ```

4.  **Make the Python wrapper available**

    Add the `py/` subdirectory to your `PYTHONPATH`. You can do this in your shell config (e.g., `~/.bashrc`, `~/.zshrc`):

    ```bash
    export PYTHONPATH=$PYTHONPATH:/jlt67/extreme-deconvolution/py
    ```



### 4. Set Up a Jupyter Notebook Kernel
To ensure the virtual environment is recognised within Jupyter notebooks, set up a kernel:
```bash
python -m ipykernel install --user --name=env --display-name "Python (Chemo-Dynamical GMM)"
```

### 5. Run the Notebooks
Open the notebooks and select the created kernel **(Python (Chemo-Dynamical GMM))** to run the code.



## For Assessment
- The associated project report can be found under [Project Report](report/report.pdf). 
- The associated executive summary can be founder under [Executive Summary](report/summary.pdf).

## License
This project is licensed under the [MIT License](https://opensource.org/license/mit/) - see the [LICENSE](LICENSE) file for details.

## Support
If you have any questions, run into issues, or just want to discuss the project, feel free to:
- Open an issue on the [GitHub Issues](https://github.com/JacobTutt/Galactic_Evol_Clustering/issues) page.  
- Reach out to me directly via [email](mailto:jacobtutt@icloud.com).

## Authors
This project is maintained by Jacob Tutt under the supervision of Dr GyuChul Myeong.



## Declaration of Use of Autogeneration Tools
This report made use of Large Language Models (LLMs) to assist in the development of the project.
These tools have been employed for:
- Formatting plots to enhance presentation quality.
- Performing iterative changes to already defined code.
- Debugging code and identifying issues in implementation.
- Helping with Latex formatting for the report.
- Identifying grammar and punctuation inconsistencies within the report.
- Helping to generate the repository's metadata files.