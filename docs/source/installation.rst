Installation and Usage
----------------------

To reproduce the analysis in these notebooks, follow the steps below:

1.  Clone the Repository

    Clone the repository to your local machine:

    .. code-block:: bash

        git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a4_coursework/jlt67.git
        cd jlt67

2.  Create a Fresh Virtual Environment

    It is recommended to use a clean virtual environment to avoid dependency conflicts:

    .. code-block:: bash

        python -m venv env
        source env/bin/activate   # For macOS/Linux
        env\Scripts\activate      # For Windows

3.  Install Dependencies

    #.  Install Python dependencies from `requirements.txt`:

        .. code-block:: bash

            pip install -r requirements.txt

    #.  Additional Installation: Extreme Deconvolution

        This project utilises the `extreme_deconvolution` algorithm for Gaussian Mixture Model (GMM) fitting, as described in Bovy et al. (2011).
        Note that this package is **not available on PyPI** and must be installed manually.

        **Source Repository**: `https://github.com/jobovy/extreme-deconvolution <https://github.com/jobovy/extreme-deconvolution>`_

        Follow these steps to install:

        #.  **Clone the Repository**

            .. code-block:: bash

               git clone https://github.com/jobovy/extreme-deconvolution.git
               cd extreme-deconvolution

        #.  **Build the C Library**

            .. code-block:: bash

               make

        #.  **Build the Python Wrapper**

            .. code-block:: bash

               make pywrapper

        #.  **Add the Python Wrapper to Your Environment**

            Add the ``py/`` subdirectory to your ``PYTHONPATH`` so it can be imported in your Python environment.
            You can do this by appending the following line to your shell configuration (e.g., ``~/.bashrc``, ``~/.zshrc``):

            .. code-block:: bash

               export PYTHONPATH=$PYTHONPATH:/jlt67/extreme-deconvolution/py

        For more detailed instructions, see the project's `README on GitHub <https://github.com/jobovy/extreme-deconvolution#readme>`_.

4.  Set Up a Jupyter Notebook Kernel

    Register the virtual environment so it's available in Jupyter:

    .. code-block:: bash

        python -m ipykernel install --user --name=env --display-name "Python (Chemo-Dynamical GMM)"

5.  Run the Notebooks

    Launch Jupyter and select the kernel named ``Python (Chemo-Dynamical GMM)`` when executing the notebooks.make html