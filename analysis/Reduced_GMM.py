import numpy as np
import time
from tabulate import tabulate
import pandas as pd

from astropy.table import Table
from astropy.io import fits
from tqdm.notebook import tqdm
import pickle as pkl

from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2, norm, gaussian_kde, multivariate_normal

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from typing import List, Tuple, Optional, Union

import umap
from sklearn.mixture import GaussianMixture


class ReducedGMMPipeline:
    """
    A pipeline for clustering stellar data using Gaussian Mixture Models (GMM) applied to UMAP-reduced feature space.

    Designed to identify and analyse structure in high-dimensional stellar datasets by first projecting them into a lower-dimensional manifold using UMAP, 
    and then clustering in this reduced space. Subsequent interpretation is performed both in the low-dimensional space and in the original parameter space.

    The pipeline follows these key steps:

    1. **Initialisation** (`__init__`):

    * Accepts stellar data in various formats: Astropy Table, NumPy recarray, or Pandas DataFrame.
    * Extracts the input features defined in `data_keys`.
    * Standardises the data and performs dimensionality reduction using UMAP (typically to 2D for visualisation and clustering).

    2. **Gaussian Mixture Model Fitting** (`run_GMM`):

    - Applies GMM clustering in the UMAP-reduced space.
    - Runs GMM for a user-defined range of component numbers and initialisations.
    - Stores log-likelihood, BIC, AIC scores, model weights, means, covariances, and predicted labels.

    3. **Model Comparison & Selection** (`compare_GMM`):

    - Compares fitted GMM models using BIC or AIC to select the optimal number of Gaussian components.
    - Allows filtering to evaluate a specific component count manually.
    - Assigns each star to a component based on the best (or filtered best) model.

    4. **Cluster Visualisation** (`plot_GMM_umap`):

    - Generates a 2D scatter plot in UMAP space, coloured by GMM cluster assignments.
    - Overlays confidence ellipses around each Gaussian component.
    - Displays marginal histograms and Gaussian curves for UMAP axes.
    - Includes a bar chart summarising the weight of each component.

    5. **High-Dimensional Interpretation** (`table_results_GMM`):

    - Computes and tabulates the mean and standard deviation of each original input feature per GMM cluster.
    - Supports custom cluster names and grouped/combined cluster analysis.
    - Helps relate low-dimensional clusters to their physical meaning in the original feature space.

    Parameters
    ----------
    star_data : Table, np.recarray, pd.DataFrame
        Input dataset containing stellar observations and features.
    data_keys : List[str]
        List of feature names (column keys) to be used for UMAP projection and back-analysis.
    error_data_keys : List[str]
        List of error feature names (column keys) corresponding to the data keys. Must be same length and order as `data_keys`.
    umap_dimensions : int
        Number of dimensions to project the data into using UMAP (default is 2).
    umap_n_neighbors : int
        Number of UMAP neighbors used for local structure preservation (default is 15).
    umap_min_dist : float
        Minimum distance between points in UMAP space; controls clustering tightness (default is 0.1).

    Attributes
    ----------
    star_data : Table
        The input dataset converted to an Astropy Table.
    feature_data : np.ndarray
        Extracted original feature values used for scaling and reference.
    feature_data_scaled : np.ndarray
        Standardised version of the original features used for UMAP.
    umap_data : np.ndarray
        Lower-dimensional representation of the data after UMAP projection.
    results_GMM : dict or None
        Stores GMM fitting results (log-likelihood, BIC/AIC scores, weights, means, covariances, labels).
    best_params : dict
        Parameters of the best GMM model selected using the chosen metric.
    filtered_best_params : dict or None
        Parameters from a user-filtered GMM model (e.g., fixed number of components).
    assignment_metric : str or None
        Indicates whether clustering assignments are from the "best" or "best filtered" model.

    Notes
    -----
    - All clustering is performed in the UMAP-reduced space.
    - Final cluster properties are summarised in both reduced and full feature spaces.
    - Supports flexible visualisation, label customisation, and component combination for interpretation.
    """
    def __init__(self, star_data: Union[Table, np.recarray, pd.DataFrame], data_keys: List[str], error_data_keys: List[str],  umap_dimensions: int = 2, umap_n_neighbors: int = 15, umap_min_dist: float = 0.1):
        """
        Initialise the XDPipeline for UMAP-based dimensionality reduction and GMM clustering.

        This method sets up the pipeline by validating the input stellar dataset, extracting specified features, 
        applying standard scaling, and reducing the feature space to a lower-dimensional UMAP representation. 
        Subsequent GMM clustering and interpretation can then be performed in this reduced space and mapped back 
        to the original feature space.

        Parameters
        ----------
        star_data : Table, np.recarray, or pd.DataFrame
            Stellar dataset containing the features of interest. Can be an Astropy Table, NumPy recarray, or Pandas DataFrame.
        data_keys : List[str]
            List of column names specifying the features to use for dimensionality reduction and clustering.
        error_data_keys : List[str]
            List of column names specifying the errors corresponding to the features in `data_keys`.
            Must be the same length and order as `data_keys`.
        umap_dimensions : int, optional
            Target number of UMAP dimensions (default is 2).
        umap_n_neighbors : int, optional
            Number of neighbors considered by UMAP for local structure (default is 15).
        umap_min_dist : float, optional
            Minimum distance between points in UMAP space (default is 0.1).

        Raises
        ------
        TypeError
            If the input dataset is not a supported type.
        ValueError
            If any of the keys in `data_keys` or 'max_gauss' are missing from the dataset.
        """

        # Convert all inputs types to an Astropy Table for consistency
        # Handels multiple input types automatically
        if isinstance(star_data, np.recarray):
            star_data = Table(star_data)  # Convert recarray to Table
        elif isinstance(star_data, pd.DataFrame):
            star_data = Table.from_pandas(star_data)  # Convert DataFrame to Table
        elif not isinstance(star_data, Table):
            raise TypeError("Unsupported data type. Must be an Astropy Table, NumPy recarray, or Pandas DataFrame.")

        # Check that the data keys and error keys are the same length
        if len(data_keys) != len(error_data_keys):
            raise ValueError("The number of data keys must match the number of error keys.")
        
        # Extract column names
        colnames = star_data.colnames
        
        # Check that the data keys and error keys are present in the data table
        # Print the missing keys if they are not present
        # We also require the 'max_gauss' key to be present in the data table - this is the results of the GMM analysis and helps us have a benchmark of the UMAP analysis preformance
        missing_keys = [key for key in data_keys + ['max_gauss'] if key not in colnames]
        if missing_keys:
            raise ValueError(f"Keys {missing_keys} not found in data table")
        

        # Store the data, data keys
        self.star_data = star_data
        self.data_keys = data_keys
        self.data_err_keys = error_data_keys
        self.umap_dimensions = umap_dimensions
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist

        # Using the UMAP parameters
        # Extract the data and their errors from the data using the keys provided and stack them into a 2D array, (number_of_parameters, number_of_samples)
        self.feature_data = np.vstack([np.asarray(self.star_data[key]) for key in self.data_keys]).T

        # Extract the errors from the astropy table using the keys provided
        self.errors_data = np.vstack([np.asarray(self.star_data[err_key]) for err_key in self.data_err_keys]).T

        # Standardise the data to have zero mean and unit variance before applying UMAP-Dimensionality Reduction
        self.scaler = StandardScaler()
        self.feature_data_scaled = self.scaler.fit_transform(self.feature_data)

        # Apply the UMAP reduction to the data with the parameters defined
        self.reducer = umap.UMAP(
            n_components=self.umap_dimensions,
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            metric='euclidean',
            random_state=10
        )
        
        # Store these umap dimensional space in the class so it can be used later
        self.umap_data = self.reducer.fit_transform(self.feature_data_scaled)

        # Extract the number of stars/ samples and features from the data array
        self.n_samples, self.n_features = self.feature_data.shape

        # Extract the labells assigned in previous analysis - ie Extreme deconvolution on full dimensional space
        self.XD_labels = self.star_data['max_gauss']

        # Initialise the following attributes for future use in the GMM pipeline
        # The range of Gaussian components to run for
        self.gauss_component_range= None
        # The maximum number of EM iterations for each GMM opitimisation
        self.max_iterations = None
        # The number of complete repetitions of the initialisations
        self.n_repeats = None
        # The number of random initialisations per component count
        self.n_init = None
        # The XD results for all runs
        self.results_GMM = None
        # The path to save the XD results for all runs
        self.save_path_GMM = None

        # The best (BIC/AIC) score and the parameters in which it was achieved
        self.best_params = {}
        # The best (BIC/AIC) score for a filtered data set and the parameters in which it was achieved
        self.filtered_best_params = None
        # Stores which of the above was used for the assignment
        self.assignment_metric = None


    def display_umap(self, label_dict: Optional[dict] = None, colour_dict: Optional[dict] = None) -> None:
        """
        Visualize the 2D UMAP projection of the data colored by cluster labels.

        Parameters
        ----------
        label_dict : dict, optional
            A mapping from numeric GMM cluster labels (e.g., 1 to 7) to string names (e.g., 'GS/E').
            If not provided, numeric labels are used directly.

        colour_dict : dict, optional
            A mapping from string names to matplotlib-compatible colors.
            Only used if label_dict is provided and names are available.
        """

        # Get numeric labels from precomputed XD or GMM labels
        labels = np.array(self.XD_labels)

        # If a mapping of labels is provided, apply it ie match cluster labels to names
        # Otherwise use raw numeric labels
        if label_dict:
            named_labels = np.array([label_dict[int(l)] for l in labels])
        else:
            named_labels = labels

        # Get unique label names
        unique_names = sorted(set(named_labels))

        # Set default colors if none are provided
        if colour_dict is None:
            default_colors = plt.cm.tab10.colors
            colour_dict = {name: default_colors[i % len(default_colors)] for i, name in enumerate(unique_names)}

        plt.figure(figsize=(8, 6))

        # Scatter plot of UMAP data colored by cluster labels
        # Iterate over unique names and plot each cluster
        for name in unique_names:
            # Create a mask for the current cluster
            mask = named_labels == name
            # Plot the UMAP data for the current cluster - colours and labeled by inputs if provided
            plt.scatter(
                self.umap_data[mask, 0],
                self.umap_data[mask, 1],
                label=name,
                color=colour_dict[name],
                s=10
            )

        plt.legend(title='Cluster', fontsize=9)
        plt.title('UMAP Projection of Stellar Data')
        plt.xlabel('UMAP-1')
        plt.ylabel('UMAP-2')
        plt.tight_layout()
        plt.show()


    def run_GMM(self, gauss_component_range: Tuple[int, int] = (1, 10), n_init: int = 10, save_path: Optional[str] = None, timings: Optional[bool] = None):
        """
        Fit Gaussian Mixture Models (GMMs) to the UMAP-reduced stellar data across a range of component numbers.

        This method fits GMMs with varying numbers of Gaussian components to the UMAP-reduced feature space. 
        For each model, it computes key metrics (log-likelihood, BIC, AIC), stores the GMM parameters 
        (weights, means, covariances), and predicts the cluster labels. Results are saved to disk if a path is provided.

        Parameters
        ----------
        gauss_component_range : Tuple[int, int], optional
            The range (min, max) of Gaussian components to try. Default is (1, 10).
        n_init : int, optional
            Number of random initialisations for each GMM. Default is 10.
        save_path : str, optional
            If provided, saves the dictionary of GMM results to this path as a pickle file.
        timings : bool, optional
            If True, returns an distionary of the average time taken for each GMM fit. (ie per component count)

        Returns
        -------
        Optional[Dict[int, float]]
            If timings is True, returns a dict of average fit times per component count.

        Raises
        ------
        ValueError
            If `gauss_component_range` is not a valid tuple of two integers, or min > max.
        TypeError
            If `n_init` is not a positive integer.

        Notes
        -----
        - GMM is applied to the UMAP-reduced data stored in `self.umap_data`.
        - The best model per component count is selected automatically using sklearn's `GaussianMixture`.
        - The results are stored in `self.results_GMM` and optionally written to disk.
        """
        # Check that the component range is valid
        if not isinstance(gauss_component_range, tuple) or len(gauss_component_range) != 2:
            raise ValueError("Gaussian component range must be a tuple of form (min, max)")
        
        if gauss_component_range[0] > gauss_component_range[1]:
            raise ValueError("Invalid gaussian component range") 
        
        # Check that the number of initialisations is valid
        if not isinstance(n_init, int) or n_init <= 0:
            raise TypeError("n_init must be a positive integer")
        

        # Save attributes to the pipeline object
        self.gauss_component_range = gauss_component_range
        self.n_init = n_init
        self.save_path_GMM = save_path


        # Initialise the results dictionary for dynamic appending
        self.results_GMM = {
            "n_gauss": [],
            "log_likelihood": [],
            "BIC": [],
            "AIC": [],
            "weights": [],
            "means": [],
            "covariances": [],
            "labels": []
        }

        # Set up timing dictionary and time tracking
        timing_dict = {}
        total_start = time.time()


        # GMM can automatically handle multiple initialisations and select the best results 
        # Therefore we do not need to worry about multiple loops for initialisations - it will simply resturn the best result
        for n_gauss in range(gauss_component_range[0], gauss_component_range[1] + 1):
            try:
                # Create a Gaussian Mixture Model with the specified number of components
                gmm = GaussianMixture(
                    n_components=n_gauss,
                    n_init=n_init,
                    covariance_type='full',
                    random_state=12
                )

                # Start the timer for fitting the GMM
                start_time_run = time.time()
                # Fit the GMM to the UMAP data
                gmm.fit(self.umap_data)

                # Record the results so they can be compared later
                log_likelihood = gmm.score(self.umap_data) * self.n_samples
                bic = gmm.bic(self.umap_data)
                aic = gmm.aic(self.umap_data)

                # This storage method allows for dynamic appending of results
                self.results_GMM["n_gauss"].append(n_gauss)
                self.results_GMM["log_likelihood"].append(log_likelihood)
                self.results_GMM["BIC"].append(bic)
                self.results_GMM["AIC"].append(aic)
                self.results_GMM["weights"].append(gmm.weights_)
                self.results_GMM["means"].append(gmm.means_)
                self.results_GMM["covariances"].append(gmm.covariances_)
                self.results_GMM["labels"].append(gmm.predict(self.umap_data))

                # End the timer for fitting the GMM
                end_time_run = time.time()

                # Average it across all of the initialisations - ie time per initialisation 
                avg_time_per_init = (end_time_run - start_time_run) / n_init
                # Save it to the timing dictionary
                timing_dict[n_gauss] = avg_time_per_init

            except Exception as e:
                print(f"GMM failed for {n_gauss} components: {e}")
            
        # Return the total run time
        total_end = time.time()
        print(f"Total run time: {total_end - total_start:.2f} seconds")

        # Save the results file to a pickle file if a save path is provided
        if save_path:
            try:
                with open(save_path, "wb") as f:
                    pkl.dump(self.results_GMM, f)
                print(f"Results saved to {save_path}")
            except Exception as e:
                print(f"Failed to save results: {e}")

        # If timings are requested, return the timing dictionary
        if timings:
            return timing_dict
        else:
            return None


    def compare_GMM(self, opt_metric='BIC', n_gauss_filter: Optional[int] = None, save_path: Optional[str] = None, display_full: bool = True, zoom_in: Optional[List[int]] = None) -> None:
        """
        Compare Gaussian Mixture Model (GMM) fits using a selected metric and assign stars to clusters.

        The method assigns stars to Gaussian components, stores  the best-fit parameters, and optionally
        visualizes the model scores.

        Parameters
        ----------
        opt_metric : str, optional
            The metric used for model selection. Must be either 'BIC' or 'AIC'. Default is 'BIC'.
        n_gauss_filter : int, optional
            If provided, only results with this number of components will be used for selection and assignment.
        save_path : str, optional
            Path to load previously saved GMM results if `self.results_GMM` is not already populated.
        display_full : bool, optional
            If True, prints a model comparison table and plots BIC/AIC scores. Default is True.
        zoom_in : List[int], optional
            A list of component numbers to zoom in on in the plot (used for inset view of BIC/AIC curves).

        Raises
        ------
        ValueError
            If the GMM results are not available and no save path is provided.
            If `opt_metric` is not 'BIC' or 'AIC'.
            If `n_gauss_filter` is out of the range of fitted components.
        """

        # Ensure that the GMM results are available either from previous runs or by loading from a file
        if self.results_GMM is None:
            if save_path is None:
                raise ValueError("No GMM results found and no save_path provided.")
            try:
                with open(save_path, 'rb') as f:
                    self.results_GMM = pkl.load(f)
            except Exception as e:
                raise ValueError(f"Failed to load results: {e}")
            
            # Determine the gaussian component range 
            self.gauss_component_range = (min(self.results_GMM["n_gauss"]), max(self.results_GMM["n_gauss"]))
        
        # Ensure input filters are valid for the GMM Results that are now accesible
        if n_gauss_filter is not None: 
            if n_gauss_filter < self.gauss_component_range[0] or n_gauss_filter > self.gauss_component_range[1]:
                raise ValueError(f"Invalid Filter: {n_gauss_filter} not in the range of Gaussian components: {self.gauss_component_range}")
            
        # Ensure valid optimisation metric is selected
        if opt_metric not in ['BIC', 'AIC']:
            raise ValueError("Invalid optimisation metric selected. Please select either 'BIC' or 'AIC'.")
        
        # Delete any existing probability assignments related entries to avoid conflicts
        colnames = self.star_data.colnames

        # Delete any existing probability assignments to avoid conflicts
        for key in colnames:
            if key.startswith('prob_gauss_umap') or key == 'max_gauss_umap':
                del self.star_data[key]

        # Extract scores and identify the best fit for the number of Gaussian components
        scores = self.results_GMM[opt_metric]
        n_gauss_all = self.results_GMM['n_gauss']
        best_idx = int(np.nanargmin(scores))

        # Store the overall best parameters - regardless of filter
        self.best_params = {
            'metric': opt_metric,
            'score': scores[best_idx],
            'gauss_components': n_gauss_all[best_idx],
            'weights': self.results_GMM['weights'][best_idx],
            'means': self.results_GMM['means'][best_idx],
            'covariances': self.results_GMM['covariances'][best_idx],
            'labels': self.results_GMM['labels'][best_idx]
        }

        if display_full:
            print(f"Best {opt_metric} Score: {self.best_params['score']:.4f}")
            print(f"  - Components: {self.best_params['gauss_components']}")


        # This preforms it on the results with a specific gaussian components filter applied
        if n_gauss_filter is not None:
            self.assignment_metric = 'best filtered'
            idx = self.results_GMM['n_gauss'].index(n_gauss_filter)
            self.filtered_best_params = {
                'filters': {"n_gauss": n_gauss_filter},
                'metric': opt_metric,
                'score': self.results_GMM[opt_metric][idx],
                'gauss_components': self.results_GMM["n_gauss"][idx],
                'weights': self.results_GMM["weights"][idx],
                'means': self.results_GMM["means"][idx],
                'covariances': self.results_GMM["covariances"][idx],
                'labels': self.results_GMM["labels"][idx]
            }

            # Handle Assignment 
            # This is much more simple in this case as GMM preforms it automaticaly for you based on filtered data
            self.star_data['max_gauss_umap'] = self.filtered_best_params['labels']
            
            # Prints summary of best performing Gaussian components
            # print the filters that can been applied
            print(f" The following filters were applied: {self.filtered_best_params['filters']}")
            # summary of results they returned
            print(f" Best {opt_metric} Score from filtered inputs: {self.filtered_best_params['score']:.4f} occurred at:")
            print(f"   - Gaussian Components (n_gauss): {self.filtered_best_params['gauss_components']}")

        else: 
            self.filtered_best_params = None
            self.assignment_metric = 'best'

            # If no filters will hndle assignment based on the best overall
            self.star_data['max_gauss_umap'] = self.best_params['labels']


        if display_full:
            # Build DataFrame for visualisation of results across different n_gauss
            df = pd.DataFrame({
                'n_gauss': self.results_GMM['n_gauss'],
                'BIC': self.results_GMM['BIC'],
                'AIC': self.results_GMM['AIC'],
                'log_likelihood': self.results_GMM['log_likelihood']
            })

            print("\nModel Fit Summary:")
            print(tabulate(df, headers='keys', tablefmt='psql'))

            # Plotting of the BIC and AIC scores over n-gauss components
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(df['n_gauss'], df['BIC'], label='BIC', color='blue', marker='o')
            ax.plot(df['n_gauss'], df['AIC'], label='AIC', color='red', marker='o')
            ax.set_xlabel('Number of Gaussian Components')
            ax.set_ylabel('Score')
            ax.set_title('GMM Model Selection Scores')
            ax.legend()
            ax.grid(True)


            # Optional zooming in functionality 
            if zoom_in:
                axins = inset_axes(ax, width='45%', height='35%', loc='upper right')
                axins.plot(df['n_gauss'], df['BIC'], 'b-o')
                axins.plot(df['n_gauss'], df['AIC'], 'r-o')
                axins.set_xlim(min(zoom_in) - 0.5, max(zoom_in) + 0.5)
                zoom_bic = [b for n, b in zip(df['n_gauss'], df['BIC']) if n in zoom_in]
                axins.set_ylim(min(zoom_bic) * 0.98, max(zoom_bic) * 1.02)
                mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='0.5')

            plt.tight_layout()
            plt.show()

    def plot_GMM_umap(self, z_score: float = 2.0, color_palette: Optional[list] = None,
                xlim: Optional[tuple] = None,
                ylim: Optional[tuple] = None) -> None:
        
        """
        Visualize GMM clustering results in the 2D UMAP-reduced space with component ellipses and marginal histograms.

        This method creates a comprehensive figure showing:
        
        - A scatter plot of stars in the 2D UMAP space colored by their GMM-assigned components.
        - Ellipses representing 2D confidence intervals (z-score-scaled) for each Gaussian component.
        - Marginal histograms for UMAP-1 and UMAP-2 projections overlaid with individual and total Gaussian fits.
        - A bar chart showing the relative weights of each Gaussian component.

        Parameters
        ----------
        z_score : float, optional
            Z-score used to scale the confidence ellipses. Default is 2.0 (~95% confidence region).
        color_palette : list, optional
            List of colors to use for the components. If None, defaults to seaborn "tab10" palette.
        xlim : tuple, optional
            Limits for the x-axis (UMAP-1). If None, inferred from data.
        ylim : tuple, optional
            Limits for the y-axis (UMAP-2). If None, inferred from data.

        Raises
        ------
        ValueError
            If no assignment metric is found. The GMM comparison must be run first.
        """

        # Ensure that the analysis has been run before generating the table
        if self.assignment_metric is None:
            raise ValueError("No assignment metric found. Please run the comparison method first.")

        # Extract the relevant parameters depending on the assignment metric used during assignment_XD
        if self.assignment_metric == 'best':
            assignment_params = self.best_params 
        elif self.assignment_metric == 'best filtered':
            assignment_params = self.filtered_best_params

        # Extract Gaussian mixture parameters
        means = assignment_params['means']
        covs = assignment_params['covariances']
        weights = assignment_params['weights']
        n_components = assignment_params['gauss_components']
        # Extract the UMAP data
        umap_data = self.umap_data
        # Extract the labels assigned to the stars
        labels = self.star_data['max_gauss_umap']

        # Color setup
        colors = color_palette if color_palette else sns.color_palette("tab10", n_components)

        # Layout of the figure
        fig = plt.figure(figsize=(8, 8))
        ax_main = plt.axes([0.1, 0.1, 0.6, 0.6])
        ax_histx = plt.axes([0.1, 0.71, 0.6, 0.19])
        ax_histy = plt.axes([0.71, 0.1, 0.19, 0.6])
        ax_bar = plt.axes([0.71, 0.71, 0.19, 0.19])

        # Main scatter of UMAP data colored by GMM labels - using the defined color palette if provided
        ax_main.scatter(umap_data[:, 0], umap_data[:, 1], c=[colors[i] for i in labels], s=3, alpha=0.5)
        ax_main.set_xlabel("UMAP-1", fontsize=19)
        ax_main.set_ylabel("UMAP-2", fontsize=19)
        ax_main.tick_params(axis='both', which='major', labelsize=18)
        if xlim: ax_main.set_xlim(xlim)
        if ylim: ax_main.set_ylim(ylim)

        # Draw ellipses on scattered data from the GMM components in 2D 
        for i in range(n_components):
            mean = means[i]
            cov = covs[i]
            eigvals, eigvecs = np.linalg.eigh(cov[:2, :2])
            order = np.argsort(eigvals)[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigvals) * np.sqrt(chi2.ppf(norm.cdf(z_score) * 2 - 1, df=2))
            ellipse = Ellipse(xy=mean[:2], width=width, height=height, angle=angle,
                            edgecolor=colors[i], facecolor='none', linewidth=1.5, alpha=0.8)
            ax_main.add_patch(ellipse)

        # Marginal Histograms of data distribution on the left and right axis
        bins_x = np.linspace(np.min(umap_data[:, 0]), np.max(umap_data[:, 0]), 40)
        bins_y = np.linspace(np.min(umap_data[:, 1]), np.max(umap_data[:, 1]), 40)
        bin_width_x = np.diff(bins_x)[0]
        bin_width_y = np.diff(bins_y)[0]
        hist_x = ax_histx.hist(umap_data[:, 0], bins=bins_x, color='gray', alpha=0.5)
        hist_y = ax_histy.hist(umap_data[:, 1], bins=bins_y, color='gray', alpha=0.5, orientation='horizontal')
        ax_histx.set_xticks([])
        ax_histy.set_yticks([])
        ax_histx.tick_params(axis='y', which='major', labelsize=18)
        ax_histy.tick_params(axis='x', which='major', labelsize=18)
        if xlim: ax_histx.set_xlim(xlim)
        if ylim: ax_histy.set_ylim(ylim)

        # Gaussian overlays on marginal histograms to show each component's contribution to the 1D axis
        x_min = min(np.min(umap_data[:, 0]), xlim[0]) if xlim else np.min(umap_data[:, 0])
        x_max = max(np.max(umap_data[:, 0]), xlim[1]) if xlim else np.max(umap_data[:, 0])
        y_min = min(np.min(umap_data[:, 1]), ylim[0]) if ylim else np.min(umap_data[:, 1])
        y_max = max(np.max(umap_data[:, 1]), ylim[1]) if ylim else np.max(umap_data[:, 1])

        x_range = np.linspace(x_min, x_max, 300)
        y_range = np.linspace(y_min, y_max, 300)

        total_gauss_x = np.zeros_like(x_range)
        total_gauss_y = np.zeros_like(y_range)

        # Calculate the Gaussian distributions for each component
        # and plot them on the marginal histograms
        for i in range(n_components):
            mx, my = means[i][:2]
            sx = np.sqrt(covs[i][0, 0])
            sy = np.sqrt(covs[i][1, 1])
            weight = weights[i]
            scale_x = np.sum(hist_x[0]) * weight * bin_width_x
            scale_y = np.sum(hist_y[0]) * weight * bin_width_y
            gauss_x = norm.pdf(x_range, mx, sx) * scale_x
            gauss_y = norm.pdf(y_range, my, sy) * scale_y
            total_gauss_x += gauss_x
            total_gauss_y += gauss_y
            ax_histx.plot(x_range, gauss_x, color=colors[i], alpha=0.7)
            ax_histy.plot(gauss_y, y_range, color=colors[i], alpha=0.7)

        ax_histx.plot(x_range, total_gauss_x, color='black', linewidth=2)
        ax_histy.plot(total_gauss_y, y_range, color='black', linewidth=2)

        # Weights bar chart for each component in the left hand side
        sorted_idx = np.argsort(weights)[::-1]
        sorted_weights = weights[sorted_idx]
        sorted_colors = [colors[i] for i in sorted_idx]
        bars = ax_bar.bar(np.arange(n_components), sorted_weights * 100,
                        color=sorted_colors, edgecolor='black', linewidth=0.8)

        for bar, w in zip(bars, sorted_weights * 100):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, w + 2, f"{w:.1f}%", ha='center',
                        va='bottom', fontsize=12, rotation=90)

        ax_bar.set_ylim(0, 75)
        ax_bar.set_xticks([])
        ax_bar.set_yticklabels([])
        ax_bar.yaxis.grid(True, linestyle='--', alpha=1)
        ax_bar.set_axisbelow(True)
        plt.show()
    
    def table_results_GMM(self, component_name_dict: dict = None, combine: list = None, labels_combined: list = None,  deconvolve: bool = False) -> pd.DataFrame:
        """
        Generate a summary table of the GMM components, projecting labels from UMAP space back to the original feature space.

        For each Gaussian component, the table reports:

        - GMM weight (%), assigned star count, and count fraction
        - Mean ± standard deviation for each feature in `self.data_keys`

        Optional:

        - Rename and reorder components using `component_name_dict`
        - Combine selected components with `combine` and `labels_combined`
        - Deconvolve observational uncertainties from the feature standard deviations using `deconvolve=True`

        Parameters
        ----------
        component_name_dict : dict, optional
            Maps component indices to custom names and defines display order.
        combine : list of list of int, optional
            List of component index groups to aggregate.
        labels_combined : list of str, optional
            Labels for each group in `combine`.
        deconvolve : bool, optional
            If True, subtracts the mean squared observational error (from `self.data_err_keys`)
            from the variance of each feature before computing the standard deviation.
            Assumes independent Gaussian errors. Requires `self.data_err_keys` to be defined
            and aligned with `self.data_keys`.

        Returns
        -------
        pd.DataFrame
            Table summarising component statistics in original feature space.

        Raises
        ------
        ValueError
            If assignments haven't been computed or `combine`/`labels_combined` lengths mismatch.
        """
        # Ensure that the analysis has been run before generating the table
        if self.assignment_metric is None:
            raise ValueError("No assignment metric found. Please run the comparison method first.")

        # Extract the relevant parameters depending on the assignment metric used during assignment_XD
        if self.assignment_metric == 'best':
            assignment_params = self.best_params 
        elif self.assignment_metric == 'best filtered':
            assignment_params = self.filtered_best_params

        # Extract Gaussian mixture parameters - ie the weights
        # Here we createa table of the GMM statisitcs in the high dimensional feature space so the actual GMM UMAP means and covariances are not used
        weights = assignment_params['weights']
        n_components = assignment_params['gauss_components']
        labels = assignment_params['labels']
        unique_components = np.unique(labels)

        # Create a DataFrame to store the results
        table_data = {
        "Component": [],
        "Weight (%)": [],
        "Count": [],
        "Count (%)": []
        }

        total_count = len(labels)

        # Cycle through the components and calculate the mean and std for each of the features in the high dimensional space
        for comp in unique_components:
            # Use the labels assigned in the low dimensional UMAP space to get the mean and std of the features in the high dimensional space
            mask = labels == comp
            count = np.sum(mask)
            count_pct = np.round(count / total_count * 100, 1)
            weight_pct = round(weights[comp] * 100, 1)

            table_data["Component"].append(f"Component {comp}")
            table_data["Weight (%)"].append(weight_pct)
            table_data["Count"].append(count)
            table_data["Count (%)"].append(count_pct)

            # For all the features in the high dimensional space, calculate the mean and std and add them to the table
            # for key in self.data_keys:
            for i in range(len(self.data_keys)):
                key = self.data_keys[i]

                values = np.array(self.star_data[key])[mask]
                mean = np.mean(values)
                std = np.std(values)

                if deconvolve:
                    # If deconvolution is used, we need to use the error keys to get the mean and std
                    error_key = self.data_err_keys[i]
                    error_values = np.array(self.star_data[error_key])[mask]
                    mean_error_squared = np.mean(error_values ** 2)
                    # Deconvolve the mean std deviation of gaussian fitted
                    var = std**2 - mean_error_squared
                    std = np.sqrt(var) if var > 0 else 0.0
                
                col = table_data.get(key, [])
                col.append(f"{mean:.2f} ± {std:.2f}")
                table_data[key] = col

        df = pd.DataFrame(table_data)

        # Rename or reorder if desired
        if component_name_dict:
            # Convert component indices to names
            df["Component"] = df["Component"].apply(lambda x: component_name_dict.get(int(x.split()[-1]), x))
            # Enforce order as it appears in the input dict
            ordered_names = list(component_name_dict.values())
            df["Component"] = pd.Categorical(df["Component"], categories=ordered_names, ordered=True)
            df = df.sort_values("Component").reset_index(drop=True)
        else:
            df = df.sort_values(by="Weight (%)", ascending=False).reset_index(drop=True)

        print("\nSummary of GMM Components (from UMAP labels + weights)")
        print(tabulate(df, headers="keys", tablefmt="grid"))

        # Optional combined row summaries
        #  this is useful for combining components that are similar ie GS/E1 and GS/E2
        if combine and labels_combined:
            if len(combine) != len(labels_combined):
                raise ValueError("combine and labels_combined must have the same length.")

            print("\nCombined Component Summary")
            combined_rows = []
            for indices, label in zip(combine, labels_combined):
                combined_mask = np.isin(labels, indices)
                count = np.sum(combined_mask)
                count_pct = round(count / total_count * 100, 1)
                weight_pct = round(np.sum([weights[i] for i in indices]) * 100, 1)

                row = {
                    "Component": label,
                    "Weight (%)": weight_pct,
                    "Count": count,
                    "Count (%)": count_pct
                }

                for i in range(len(self.data_keys)):
                    key = self.data_keys[i]

                    values = np.array(self.star_data[key])[combined_mask]
                    mean = np.mean(values)
                    std = np.std(values)

                    if deconvolve:
                        # If deconvolution is used, we need to use the error keys to get the mean and std
                        error_key = self.data_err_keys[i]
                        error_values = np.array(self.star_data[error_key])[combined_mask]
                        mean_error_squared = np.mean(error_values ** 2)
                        # Deconvolve the mean std deviation of gaussian fitted
                        var = std**2 - mean_error_squared
                        std = np.sqrt(var) if var > 0 else 0.0
                        
                
                    row[key] = f"{mean:.2f} ± {std:.2f}"

                combined_rows.append(row)

            combined_df = pd.DataFrame(combined_rows)
            print(tabulate(combined_df, headers="keys", tablefmt="grid"))

        return df

    def plot_highdim_gaussian(self, x_key: str, y_key: str, z_score: float = 2.0,
                full_survey_file: Optional[str] = None,
                color_palette: Optional[list] = None,
                xlim: Optional[tuple] = None,
                ylim: Optional[tuple] = None, 
                deconvolve: bool = False, 
                legend: Optional[tuple] = None) -> None:
        """
        Visualize GMM component assignments in high-dimensional space for two selected features.

        Generates a 2D scatter plot of stars colored by GMM component, with:

        - Confidence ellipses estimated from the empirical mean and covariance of each component
        - Marginal histograms with overlaid Gaussian fits
        - Optional background 2D histogram from a reference survey
        - Top-right bar chart showing GMM component weights

        Parameters
        ----------
        x_key : str
            Name of the column to plot on the x-axis.
        y_key : str
            Name of the column to plot on the y-axis.
        z_score : float, optional
            Controls the confidence interval of Gaussian ellipses. Default is 2.0 (~95%).
        full_survey_file : str, optional
            Path to FITS file for background sample (e.g., full survey for density plot).
        color_palette : list, optional
            List of custom colors for components. If None, uses Seaborn's 'husl' palette.
        xlim : tuple, optional
            x-axis limits (min, max).
        ylim : tuple, optional
            y-axis limits (min, max).
        deconvolve : bool, optional
            If True, subtracts the mean squared observational errors (from `self.data_err_keys`)
            from the empirical covariance matrix of each component before plotting the ellipses.
            This reveals the intrinsic spread of each GMM component, assuming independent
            Gaussian measurement errors in x and y.
        legend : tuple, optional
            If provided, a tuple of (x, y) coordinates for the legend position.
        

        Raises
        ------
        ValueError
            If the assignment method hasn't been run or if input keys aren't found in the data table.
        """

        # Ensures analysis has been run before plotting
        if self.assignment_metric is None:
            raise ValueError("No assignment metric found. Please run assignment method first.")

        # Ensures that the keys provided are valid
        if x_key not in self.star_data.colnames or y_key not in self.star_data.colnames:
            raise ValueError("Invalid keys. Please provide valid keys from the data table (i.e., exist in data_keys).")

        # Extracts the relevant parameters depending on the assignment metric used during assignment_XD
        if self.assignment_metric == 'best':
            assignment_params = self.best_params 
        elif self.assignment_metric == 'best filtered':
            assignment_params = self.filtered_best_params

        # Extracts GMM parameters from the relevant best-fit analysis
        weights = assignment_params['weights']
        n_components = assignment_params['gauss_components']

        # Retrieves column index directly from the star_data table
        x_index = self.data_keys.index(x_key)
        y_index = self.data_keys.index(y_key)

        if deconvolve:
            # Extract relavent error keys - which are the same order as the data keys
            x_error_key = self.data_err_keys[x_index]
            y_error_key = self.data_err_keys[y_index]

        # Extracts the individual star data
        x_data = np.asarray(self.star_data[x_key])
        y_data = np.asarray(self.star_data[y_key])
        assignments = assignment_params['labels']

        # Fit the data to a 2D Gaussian rather than using the XD results 
        # Convert to 2D array for easier indexing
        xy_data = np.vstack([x_data, y_data]).T
        # Initialize list to store calculated ellipses
        ellipse_params = []
        for i in range(0, n_components):
            # Select data for component i
            cluster_points = xy_data[assignments == i]
            if deconvolve:
                # Extract the error data for the relevant component
                x_error_data = np.asarray(self.star_data[x_error_key])[assignments == i]
                y_error_data = np.asarray(self.star_data[y_error_key])[assignments == i]

                # Work out the 2x2 covariance matrix for the errors
                var_x_err = np.mean(x_error_data ** 2)
                var_y_err = np.mean(y_error_data ** 2)

                error_cov = np.array([[var_x_err, 0], [0, var_y_err]])

            if len(cluster_points) < 2:
                continue  # Skip components with insufficient points
            # Compute mean and covariance
            mean = np.mean(cluster_points, axis=0)
            cov = np.cov(cluster_points, rowvar=False)
            # If deconvolution is used, we need to add the error covariance to the data covariance
            if deconvolve:
                # Add the remove the data covariance
                cov -= error_cov

            ellipse_params.append((mean, cov))

        # Set axis labels (custom formatting) - Use scaled or unscaled labels based on the scaling flag
        axis_label_dict = {
            'fe_h': r'[Fe/H]', 'E_50': r'Energy', 'Energy': r'Energy',
            'alpha_m': r'[$\alpha$/Fe]', 'alpha_fe': r'[$\alpha$/Fe]', 'ce_fe': r'[Ce/Fe]',
            'al_fe': r'[Al/Fe]', 'Al_fe': r'[Al/Fe]', 'mg_mn': r'[Mg/Mn]',
            'Y_fe': r'[Y/Fe]', 'Mg_Mn': r'[Mg/Mn]', 'Mn_fe': r'[Mn/Fe]',
            'Ba_fe': r'[Ba/Fe]', 'Mg_Cu': r'[Mg/Cu]', 'Eu_fe': r'[Eu/Fe]',
            'Ba_Eu': r'[Ba/Eu]', 'Na_fe': r'[Na/Fe]', 'Ni_fe': r'[Ni/Fe]', 'Eu_Mg': r'[Eu/Mg]', 'ni_fe': r'[Ni/Fe]'
        }

        xlabel = axis_label_dict.get(x_key, x_key)
        ylabel = axis_label_dict.get(y_key, y_key)

        # Defines colors for different components
        colors = color_palette if color_palette else sns.color_palette("husl", n_components)

        # Defines grid for subplots 
        fig = plt.figure(figsize=(8, 8))
        ax_main = plt.axes([0.1, 0.1, 0.6, 0.6])
        ax_histx = plt.axes([0.1, 0.71, 0.6, 0.19])
        ax_histy = plt.axes([0.71, 0.1, 0.19, 0.6])
        ax_bar = plt.axes([0.71, 0.71, 0.19, 0.19])

        # Background survey data
        if full_survey_file:
            with fits.open(full_survey_file) as hdul:
                survey_data = hdul[1].data

            # Capitalize only if file is from Apogee
            if "Apogee" in full_survey_file or "APOGEE" in full_survey_file:
                x_key_lookup = x_key.upper()
                y_key_lookup = y_key.upper()
            else:
                x_key_lookup = x_key
                y_key_lookup = y_key

            if x_key_lookup in survey_data.columns.names and y_key_lookup in survey_data.columns.names:
                full_x = survey_data[x_key_lookup]
                full_y = survey_data[y_key_lookup]

                # Remove NaNs
                mask = ~np.isnan(full_x) & ~np.isnan(full_y)

                # Plot background density
                ax_main.hist2d(
                    full_x[mask], full_y[mask],
                    bins=400, cmap='Greys', cmin=0, alpha=1,
                    norm=mcolors.PowerNorm(gamma=0.5)
                )

        # Main scatter plot - with stars coloured by their assigned Gaussian component
        ax_main.scatter(x_data, y_data, c=[colors[i] for i in assignments], s=3, alpha=0.5)
        ax_main.set_xlabel(xlabel, fontsize=19)
        ax_main.set_ylabel(ylabel, fontsize=19)
        ax_main.tick_params(axis='both', which='major', labelsize=18)

        if xlim:
            ax_main.set_xlim(xlim)
        if ylim:
            ax_main.set_ylim(ylim)

        # Plots Gaussian components as ellipses
        for i, (mean, cov) in enumerate(ellipse_params):
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigvals)

            scale = np.sqrt(chi2.ppf(norm.cdf(z_score) * 2 - 1, df=2))
            width *= scale
            height *= scale

            lw = np.clip(1.0 + 4.0 * weights[i], 1.0, 3.5)
            alpha = np.clip(0.4 + 0.7 * weights[i], 0.4, 0.9)

            # alpha_array = [0.75, 1, 1, 0.75, 1, 1]
            # alpha = alpha_array[i]
            ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                            edgecolor=colors[i], facecolor='none', linewidth=lw, alpha=alpha, linestyle='--')
            ax_main.add_patch(ellipse)

        if y_key == 'E_50' or y_key == 'Energy':
            ax_main.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 1e-5:.1f}"))
            ax_main.set_ylabel(f"{ylabel} ($\\times 10^5$)", fontsize=18)

        # Histograms
        bins_x = np.linspace(np.min(x_data), np.max(x_data), 40)
        bins_y = np.linspace(np.min(y_data), np.max(y_data), 40)
        bin_width_x = np.diff(bins_x)[0]
        bin_width_y = np.diff(bins_y)[0]
        hist_x, _, _ = ax_histx.hist(x_data, bins=bins_x, color='gray', alpha=0.5)
        hist_y, _, _ = ax_histy.hist(y_data, bins=bins_y, color='gray', alpha=0.5, orientation='horizontal')
        ax_histx.set_xticks([])
        ax_histy.set_yticks([])
        if xlim:
            ax_histx.set_xlim(xlim)
        if ylim:
            ax_histy.set_ylim(ylim)
        ax_histx.tick_params(axis='y', which='major', labelsize=18)
        ax_histy.tick_params(axis='x', which='major', labelsize=18)

        # Gaussian overlays
        x_min = min(np.min(x_data), xlim[0]) if xlim else np.min(x_data)
        x_max = max(np.max(x_data), xlim[1]) if xlim else np.max(x_data)
        y_min = min(np.min(y_data), ylim[0]) if ylim else np.min(y_data)
        y_max = max(np.max(y_data), ylim[1]) if ylim else np.max(y_data)

        x_range = np.linspace(x_min, x_max, 300)
        y_range = np.linspace(y_min, y_max, 300)
        total_gauss_x = np.zeros_like(x_range)
        total_gauss_y = np.zeros_like(y_range)
        
        for i, (mean, cov) in enumerate(ellipse_params):
            mean_x, mean_y = mean[0], mean[1]
            std_x = np.sqrt(cov[0, 0])
            std_y = np.sqrt(cov[1, 1])
            weight = weights[i]

            scale_x = np.sum(hist_x) * weight * bin_width_x
            scale_y = np.sum(hist_y) * weight * bin_width_y
            gauss_x = norm.pdf(x_range, mean_x, std_x) * scale_x
            gauss_y = norm.pdf(y_range, mean_y, std_y) * scale_y
            total_gauss_x += gauss_x
            total_gauss_y += gauss_y
            ax_histx.plot(x_range, gauss_x, color=colors[i], alpha=0.7,  ls ='--')
            ax_histy.plot(gauss_y, y_range, color=colors[i], alpha=0.7,  ls ='--')

        ax_histx.plot(x_range, total_gauss_x, color='black', linewidth=2, ls ='--')
        ax_histy.plot(total_gauss_y, y_range, color='black', linewidth=2, ls ='--')

        # TOP RIGHT - Bar chart of relative weights
        sorted_indices = np.argsort(weights)[::-1]
        sorted_weights = weights[sorted_indices]
        sorted_colors = [colors[i] for i in sorted_indices]
        bar_heights = sorted_weights * 100
        bar_positions = np.arange(n_components)

        bars = ax_bar.bar(bar_positions, bar_heights, width=0.8, color=sorted_colors,
                        edgecolor='black', linewidth=0.8)

        for bar, weight in zip(bars, bar_heights):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, weight + 2,
                        f"{weight:.1f}%", ha='center', va='bottom',
                        fontsize=13, color='black', rotation=90)

        ax_bar.set_ylim(0, 75)
        ax_bar.set_xlim(-0.5, n_components - 0.5)
        ax_bar.set_yticks([0, 20, 40, 60])
        ax_bar.set_yticklabels([])
        ax_bar.set_xticks([])
        ax_bar.yaxis.grid(True, linestyle='--', alpha=1)
        ax_bar.set_axisbelow(True)

        # Add an optional legend
        if legend:
            legend_elements = []
            for idx, name in legend.items():
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', label=name,
                        markerfacecolor=colors[idx], markersize=8, alpha=0.7)
                )
            ax_main.legend(
                handles=legend_elements,
                loc='upper right',
                fontsize=15,
                frameon=True,
                facecolor='white'
    )

        plt.show()