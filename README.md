# Chemo-dynamical analysis of Milky Way’s stellar populations with unsupervised multi-dimensional clustering
## Supervisor: Dr GyuChul Myeong, Institute of Astronomy

This repository aims to reproduce and build upon the paper:  `Milky Way’s Eccentric Constitutents with Gaia, APOGEE, and
GALAH`, Myeong 2022

---

# Installation Instructions

To run the notebooks, please follow these steps:

## 1. Clone the Repository

Clone the repository from the remote repository (GitLab) to your local machine.

```bash
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a4_coursework/jlt67.git
cd jlt67
```

## 2. Create a Fresh Virtual Environment
Use a clean virtual environment to avoid dependency conflicts.
```bash
python -m venv env
source env/bin/activate   # For macOS/Linux
env\Scripts\activate      # For Windows
```

## 3. Install the dependencies
### 3.1 Navigate to the repository’s root directory and install the PyPi package dependencies:
```bash
cd jlt67
pip install -r requirements.txt
```
### 3.2 Additional Installation of Extreme Deconvolution

This project utilizes the `extreme_deconvolution` algorithm for Gaussian Mixture Model (GMM) implementation. This provided by Bovy et al. (2011) but is not directly available via PyPI.

To install the package, follow these steps, or see the source repositories `README`:

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



## 4. Set Up a Jupyter Notebook Kernel
To ensure the virtual environment is recognised within Jupyter notebooks, set up a kernel:
```bash
python -m ipykernel install --user --name=env --display-name "Python (Chemo-Dynamical GMM)"
```

## 5. Run the Notebooks
Open the notebooks and select the created kernel **(Python (Chemo-Dynamical GMM))** to run the code.



## For Assessment

### Report
Please find the projects report under `Report` directory

### Declaration of Use of Autogeneration Tools
This report made use of Large Language Models (LLMs) to assist in the development of the project.
These tools have been employed for:
- Formatting plots to enhance presentation quality.
- Performing iterative changes to already defined code.
- Debugging code and identifying issues in implementation.
- Helping with Latex formatting for the report.
- Identifying grammar and punctuation inconsistencies within the report.