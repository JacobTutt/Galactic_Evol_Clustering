import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd

from astropy.table import Table
from tqdm.notebook import tqdm
import pickle as pkl

from extreme_deconvolution import extreme_deconvolution
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal

from typing import List, Tuple, Optional, Union

class XDPipeline:
    """
    """
    def __init__(self, star_data: Union[Table, np.recarray, pd.DataFrame], data_keys: List[str], data_err_keys: List[str]):
        """
        Initialise the XDPipeline with stellar data and keys of intrest for the Gaussian Mixture Model (GMM) - Extreme Deconvolution (XD) process, defining the parameter space of interest. 

        Parameters:
            star_data (Table, np.recarray, pd.DataFrame): Dataset containing stellar information, which can be an Astropy Table, NumPy recarray, or Pandas DataFrame.
            data_keys (List[str]): List of column names representing features used in the GMM fitting.
            data_err_keys (List[str]): List of column names representing measurement uncertainties, corresponding to `data_keys`.

        Raises:
            TypeError: If the input dataset is not a supported type.
            ValueError: If `data_keys` and `data_err_keys` have mismatched lengths or contain missing columns.
        """

        # Entry checks
        # Check if data has column names and if so extract them accordingly
        if hasattr(star_data, 'colnames'):  # For Astropy Table
            colnames = star_data.colnames
        elif hasattr(star_data, 'dtype') and hasattr(star_data.dtype, 'names'):  # For recarray (FITS)
            colnames = star_data.dtype.names
        elif isinstance(star_data, pd.DataFrame):  # For Pandas DataFrame
            colnames = star_data.columns
        else:
            raise TypeError("Unsupported data type. Must have 'colnames', 'dtype.names' attribute, or be a Pandas DataFrame.")
        
        # Check that the number of data keys and error keys match
        if len(data_keys) != len(data_err_keys):
            raise ValueError("Size of data and error keys must match")
        
        # Check that the data keys and error keys are present in the data table
        # Print the missing keys if they are not present
        missing_keys = [key for key in data_keys + data_err_keys if key not in colnames]
        if missing_keys:
            raise ValueError(f"Keys {missing_keys} not found in data table")

        # Store the data, data keys, and data error keys
        self.star_data = star_data
        self.data_keys = data_keys
        self.data_err_keys = data_err_keys

        # Extract the data and their errors from the data using the keys provided and stack them into a 2D array, (number_of_parameters, number_of_samples)
        self.feature_data = np.vstack([self.star_data[key] for key in self.data_keys]).T

        # Extract the errors from the astropy table using the keys provided
        # We have no informatiom on the correlation between the errors so assume they are diagonal/uncorrelated. 
        self.errors_data = np.vstack([self.star_data[err_key] for err_key in self.data_err_keys]).T

        # Extract the number of stars/ samples and features from the data array
        self.n_samples, self.n_features = self.feature_data.shape

        # Initialise the following attributes for future use in the XD pipeline
        # The range of Gaussian components to run for
        self.gauss_component_range= None
        # The maximum number of EM iterations for each GMM opitimisation
        self.max_iterations = None
        # The number of complete repetitions of the initialisations
        self.n_repeats = None
        # The number of random initialisations per component count
        self.n_init = None
        # The XD results for all runs
        self.results_XD = None
        # The path to save the XD results for all runs
        self.save_path_XD = None

        # The best (BIC/AIC) score and the parameters in which it was achieved
        self.best_params = {}
        # The best (BIC/AIC) score for a filtered data set and the parameters in which it was achieved
        self.filtered_best_params = None




    def _BICScore(self, log_likelihood: float, num_params: int, num_data_points: int) -> float:
        """
        Compute the Bayesian Information Criterion (BIC) score.

        Parameters
        ----------
        log_likelihood : float
            Log-likelihood of the model.
        num_params : int
            Number of free parameters in the model.

        Returns
        -------
        float
            Computed BIC score.
        """
        return -2 * log_likelihood + num_params * np.log(num_data_points)
    
    def _AICScore(self, log_likelihood: float, num_params: int) -> float:
        """
        Compute the Akaike Information Criterion (AIC) score.

        Parameters
        ----------
        log_likelihood : float
            Log-likelihood of the model.
        num_params : int
            Number of free parameters in the model.

        Returns
        -------
        float
            Computed AIC score.
        """
        return - 2 * log_likelihood + 2 * num_params
    
    def run_XD(self, gauss_component_range: Tuple[int, int] = (1, 10), max_iterations: int = int(1e9), n_repeats: int = 3, n_init: int = 100, save_path: Optional[str] = None) -> None:
        """
        Run Extreme Deconvolution (XD) for a range of Gaussian components with multiple random initializations.
        The results are evaluated using BIC and AIC for model selection.

        Parameters
        ----------
        gauss_component_range : Tuple[int, int]
            Range of Gaussian components to test, specified as (min, max).
        max_iterations : int
            Maximum number of EM iterations per XD run.
        n_repeats : int
            Number of complete repetitions of the initialization process.
        n_init : int
            Number of random initializations per component count.
        save_path : Optional[str]
            Path to save XD results. If None, results are not saved.

        Raises
        ------
        ValueError
            If `gauss_component_range` is invalid.
        TypeError
            If `n_repeats`, `n_init`, or `max_iterations` are not positive integers.
        """

        # Check that the component range is valid
        if not isinstance(gauss_component_range, tuple) or len(gauss_component_range) != 2:
            raise ValueError("Gaussian component range must be a tuple of form (min, max)")
        
        if gauss_component_range[0] > gauss_component_range[1]:
            raise ValueError("Invalid gaussian component range") 
        
        # Check that the number of repeats and initialisations are valid
        if not isinstance(n_repeats, int) or n_repeats <= 0:
            raise TypeError("n_repeats must be a positive integer")
        if not isinstance(n_init, int) or n_init <= 0:
            raise TypeError("n_init must be a positive integer")
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise TypeError("max_iterations must be a positive integer")
        
        # Save attributes to the pipeline object
        self.gauss_component_range = gauss_component_range
        self.max_iterations = max_iterations
        self.n_repeats = n_repeats
        self.n_init = n_init
        self.save_path_XD = save_path


        # Scale the data to have zero mean and unit variance - this will improve the convergence of the EM algorithm
        # Errors are scaled by the same factor to maintain the same relative uncertainty
        # Note this will require the gaussians means and covariances returned by XD to be scaled back to the original units for interpretation before saving
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(self.feature_data)
        errors_scaled = self.errors_data / scaler.scale_

        # Calculate the extreme values of the data for initialisation randomisation
        # extreme_data_values = (np.max(self.feature_data, axis=0), np.min(self.feature_data, axis=0))
        # THIS MAY IN PART BE WHERE THE ERROR IS COMING FROM - THEY HAVE BEEN NORMALISED SO THE INITIALSIATION VALUES SHOULD BE NORMALISED TOO
        scaled_min = np.min(data_scaled, axis=0)
        scaled_max = np.max(data_scaled, axis=0)

        # Initialise the results dictionary for dynamic appending
        self.results_XD = {
            # Repeat Number
            "repeat no.": [],
            # Intialisation Number
            "init no.": [],
            # Number of gaussians fitted
            "n_gauss": [],
            # Log likelihood of the best-fit model
            "log_likelihood": [],
            # Bayesian Information Criterion
            "BIC": [],
            # Akaike Information Criterion
            "AIC": [],
            # Best-fit weights of the gaussians
            "weights": [],
            # Best-fit means of the gaussians
            "means": [],
            # Best-fit covariances of the gaussians
            "covariances": []
        }

        # Iterate for a test range of number of gaussians
        for n_gauss in tqdm(range(self.gauss_component_range[0], self.gauss_component_range[1] + 1), desc="Number of Gaussian Components"):
            # Overall repeats 
            for n in tqdm(range(self.n_repeats), desc="Repeat Cycles", leave=False):
                # Random initialisations of input parameters
                for i in tqdm(range(self.n_init), desc="Initialisations", leave=False):

                    # Even initialisation of weights
                    #init_weights = np.ones(n_gauss) / n_gauss
                    init_weights = np.random.dirichlet(np.ones(n_gauss))

                    # Data has been scaled so means should be sampled from the scaled values.
                    
                    # Randomly Initialise means from a normal distribution with mean = 0 and std = 1
                    # init_mean = np.random.normal(loc=0, scale=1, size=(n_gauss, self.n_features))
                    
                    # Randomly initialise means from the scaled values range
                    init_mean = np.random.uniform(low=scaled_min, high=scaled_max, size=(n_gauss, self.n_features))

                    # Covariances initialised as identity matrices
                    init_covar = np.array([np.identity(self.n_features) for _ in range(n_gauss)])
                    # Run XD
                    try:
                        XD_avg_LL = extreme_deconvolution(
                            data_scaled, errors_scaled, init_weights, init_mean, init_covar, maxiter=max_iterations)
                        # Calculate the total log likelihood
                        total_LL = XD_avg_LL * self.n_samples

                        # Calculate the bic and aic scores
                        num_params = n_gauss * (1 + self.n_features + self.n_features * (self.n_features + 1) // 2) - 1
                        bic, aic = self._BICScore(total_LL, num_params, self.n_samples), self._AICScore(total_LL, num_params)

                        # Copy the updated weights, means and covariances
                        post_XD_weights, post_XD_means, post_XD_cov = init_weights.copy(), init_mean.copy(), init_covar.copy()

                        # Unscale the means and covariances to return them to their original/meaningful units
                        post_scaling_means = scaler.inverse_transform(post_XD_means)
                        post_scaling_cov = np.array([
                            np.dot(np.dot(np.diag(scaler.scale_), cov), np.diag(scaler.scale_))
                            for cov in post_XD_cov
                        ])

                        # Store the results
                        self.results_XD["repeat no."].append(n)
                        self.results_XD["init no."].append(i)
                        self.results_XD["n_gauss"].append(n_gauss)
                        self.results_XD["log_likelihood"].append(total_LL)
                        self.results_XD["BIC"].append(bic)
                        self.results_XD["AIC"].append(aic)
                        self.results_XD["weights"].append(post_XD_weights)
                        self.results_XD["means"].append(post_scaling_means)
                        self.results_XD["covariances"].append(post_scaling_cov)


                    except Exception as e:
                        print(f"XD failed for {n_gauss} components, on repeat: {n}, iteration: {i}: {e}")

                        # Store the results
                        self.results_XD["repeat no."].append(n)
                        self.results_XD["init no."].append(i)
                        self.results_XD["n_gauss"].append(n_gauss)
                        self.results_XD["log_likelihood"].append(None)
                        self.results_XD["BIC"].append(None)
                        self.results_XD["AIC"].append(None)
                        self.results_XD["weights"].append(None)
                        self.results_XD["means"].append(None)
                        self.results_XD["covariances"].append(None)

            # Save the results if a path is provided
            # This is redone for each component count to ensure that the results are saved in case of a crash
            if self.save_path_XD:
                try:
                    with open(self.save_path_XD, "wb") as f:
                        pkl.dump(self.results_XD, f)
                    print(f"Results saved successfully at {self.save_path_XD}")
                except Exception as e:
                    print(f"Failed to save results: {e}")

        return None
    
    def compare_XD(self, opt_metric = 'BIC', n_gauss_filter: Optional[int] = None, repeat_no_filter: Optional[int] = None, save_path: Optional[str] = None) -> None:
        """
        Analyse Extreme Deconvolution (XD) results using BIC or AIC.
        This method identifies the best-fit model, summarizes failed runs, and visualizes score distributions.
        If no filters are applied, the analysis is performed on all results. Otherwise, it is performed on filtered results.

        Parameters
        ----------
        opt_metric : str
            Optimization metric ('BIC' or 'AIC').
        n_gauss_filter : Optional[int]
            Specific number of Gaussian components to filter results by.
        repeat_no_filter : Optional[int]
            Specific repeat cycle to filter results by.
        save_path : Optional[str]
            Path to load XD results if not already stored in the class.

        Raises
        ------
        ValueError
            If results are not available in the class and no valid `save_path` is given.
        ValueError
            If `opt_metric` is not 'BIC' or 'AIC'.
        ValueError
            If filter values (`n_gauss_filter`, `repeat_no_filter`) are outside valid ranges.
        """

        
        if self.results_XD is None:
            if save_path is None:
                raise ValueError("No XD results found in class and no path given to load from. Please run XD first or provide valid path to load from.")

            else:
                try:
                    self.save_path_XD = save_path
                    with open(self.save_path_XD, "rb") as f:
                        self.results_XD = pkl.load(f)
                except Exception as e:
                    print(f"Failed to save results: {e}")
                    raise ValueError("Failed to load results from the given path.")

            # Extract information from file data that will not be saved to the class if XD is not run in this instance

            self.gauss_component_range = (min(self.results_XD["n_gauss"]), max(self.results_XD["n_gauss"]))
            self.n_repeats = max(self.results_XD["repeat no."]) + 1
            self.n_init = max(self.results_XD["init no."]) + 1

        # Ensure input filters are valid for the XD Results that the class contains
        if n_gauss_filter is not None: 
            if n_gauss_filter < self.gauss_component_range[0] or n_gauss_filter > self.gauss_component_range[1]:
                raise ValueError(f"Invalid Filter: {n_gauss_filter} not in the range of Gaussian components: {self.gauss_component_range}")
        if repeat_no_filter is not None:
            if repeat_no_filter > self.n_repeats:
                raise ValueError(f"Invalid Filter: {repeat_no_filter} not in the range of repeat cycles: {self.n_repeats}")
        
        # Ensure valid optimisation metric is selected
        if opt_metric not in ['BIC', 'AIC']:
            raise ValueError("Invalid optimisation metric selected. Please select either 'BIC' or 'AIC'.")
        
        # Delete any columns in self.star_data that are already present from past analysis
        # Search for keys

        # Delete any existing probability assignments to avoid conflicts
        for key in list(self.star_data.keys()):
            if key.startswith('prob_gauss_') or key == 'max_gauss':
                del self.star_data[key]
        

        # This is preformed for all results ie all n_gauss, all repeats and all initialisations
        # Identifies the best BIC/AIC score - its index, correlated inputs and results
        self.best_params['metric'] = opt_metric
        self.best_params['score'] = min(b for b in self.results_XD[opt_metric] if b is not None)
        best_index = self.results_XD[opt_metric].index(self.best_params['score'])
        self.best_params['gauss_components'] = self.results_XD["n_gauss"][best_index]
        self.best_params['repeat'] = self.results_XD["repeat no."][best_index]
        self.best_params['init'] = self.results_XD["init no."][best_index]
        self.best_params['XD_weights'] = self.results_XD["weights"][best_index]
        self.best_params['XD_means'] = self.results_XD["means"][best_index]
        self.best_params['XD_covariances'] = self.results_XD["covariances"][best_index]

        # Prints summary of best performing Gaussian components
        print(f" Best Overall {opt_metric} Score: {self.best_params['score']:.4f} occurred at:")
        print(f"   - Gaussian Components (n_gauss): {self.best_params['gauss_components']}")
        print(f"   - Repeat cycle (n): {self.best_params['repeat']}")
        print(f"   - Initialisation (i): {self.best_params['init']}")

        # This preforms it on the filtered data - ie a specific n_gauss and/ or repeat cycle
        if n_gauss_filter is not None or repeat_no_filter is not None:

            # Filter the dictionary and only keep relevent results
            mask = np.ones(len(self.results_XD["n_gauss"]), dtype=bool)
            if n_gauss_filter is not None:
                mask &= (np.array(self.results_XD["n_gauss"]) == n_gauss_filter)
            if repeat_no_filter is not None:
                mask &= (np.array(self.results_XD["repeat no."]) == repeat_no_filter)

            filtered_results = {key: np.array(self.results_XD[key])[mask].tolist() for key in self.results_XD.keys()}
            # Filter the dictionary and only keep relevent results

            # Identifies the best BIC/AIC score
            self.filtered_best_params = {}
            self.filtered_best_params['filters'] = {"n_gauss": n_gauss_filter, "repeat": repeat_no_filter}
            self.filtered_best_params['metric'] = opt_metric 
            self.filtered_best_params['score'] = min(b for b in filtered_results[opt_metric] if b is not None)
            filted_index = filtered_results[opt_metric].index(self.filtered_best_params['score'])
            self.filtered_best_params['gauss_components'] = filtered_results["n_gauss"][filted_index]
            self.filtered_best_params['repeat'] = filtered_results["repeat no."][filted_index]
            self.filtered_best_params['init'] = filtered_results["init no."][filted_index]
            self.filtered_best_params['XD_weights'] = filtered_results["weights"][filted_index]
            self.filtered_best_params['XD_means'] = filtered_results["means"][filted_index]
            self.filtered_best_params['XD_covariances'] = filtered_results["covariances"][filted_index]
            
            # Prints summary of best performing Gaussian components
            # print the filters that can been applied
            print(f" The following filters were applied: {self.filtered_best_params['filters']}")
            # summary of results they returned
            print(f" Best {opt_metric} Score from filtered inputs: {self.filtered_best_params['score']:.4f} occurred at:")
            print(f"   - Gaussian Components (n_gauss): {self.best_params['gauss_components']}")
            print(f"   - Repeat cycle (n): {self.best_params['repeat']}")
            print(f"   - Initialisation (i): {self.best_params['init']}")

        else: 
            self.filtered_best_params = None
            print("No analysis was performed on filtered data as no filters inputed")


        # Analysis on overall code - not filtered
        n_gauss_list = np.array([n for n in range(self.gauss_component_range[0], self.gauss_component_range[1] + 1)])

        # Creates a DataFrame summarising the number of failed XD runs per Gaussian component
        n_XD_failed = [
            sum([b is None for c, b in zip(self.results_XD["n_gauss"], self.results_XD["BIC"]) if c == n_gauss])
            for n_gauss in n_gauss_list
        ]
        n_runs_gauss = np.array([self.n_repeats * self.n_init for _ in n_gauss_list])
        n_failed_XD = pd.DataFrame({
            "No. Gaussians": n_gauss_list,
            "No. Failed XD runs": n_XD_failed,
            "Total No. Runs": n_runs_gauss
        })

        # Prints a formatted table of the number of failed XD runs
        print("Table of Number of Gaussians vs Number of Failed XD Runs")
        print(tabulate(n_failed_XD, headers='keys', tablefmt='psql'))


        # Reshapes BIC and AIC scores to 3D arrays: (n_gauss_components, n_repeats, n_init)
        BIC_scores = np.array([b if b is not None else np.nan for b in self.results_XD['BIC']]).reshape((len(n_gauss_list), self.n_repeats, self.n_init))
        AIC_scores = np.array([a if a is not None else np.nan for a in self.results_XD['AIC']]).reshape((len(n_gauss_list), self.n_repeats, self.n_init))

        # Compute means and standard deviations for BIC and AIC across initialisations
        BIC_means, BIC_stds = np.nanmean(BIC_scores, axis=2), np.nanstd(BIC_scores, axis=2)
        AIC_means, AIC_stds = np.nanmean(AIC_scores, axis=2), np.nanstd(AIC_scores, axis=2)

        # Format mean ± stddev for BIC and AIC into DataFrames
        BIC_means_stds_df = pd.DataFrame(
            np.array([[f"{mean:.5f} \\pm {std:.5f}" for mean, std in zip(mean_row, std_row)]
                    for mean_row, std_row in zip(BIC_means, BIC_stds)]),
            columns=[f"Repeat {i + 1}" for i in range(self.n_repeats)],
            index=n_gauss_list
        )
        BIC_means_stds_df.insert(0, "No. Gaussians", n_gauss_list)

        AIC_means_stds_df = pd.DataFrame(
            np.array([[f"{mean:.5f} \\pm {std:.5f}" for mean, std in zip(mean_row, std_row)]
                    for mean_row, std_row in zip(AIC_means, AIC_stds)]),
            columns=[f"Repeat {i + 1}" for i in range(self.n_repeats)],
            index=n_gauss_list
        )
        AIC_means_stds_df.insert(0, "No. Gaussians", n_gauss_list)

        # Print BIC and AIC summary tables
        print("Table of BIC Means and Stds")
        print(tabulate(BIC_means_stds_df, headers='keys', tablefmt='psql'))

        print("Table of AIC Means and Stds")
        print(tabulate(AIC_means_stds_df, headers='keys', tablefmt='psql'))

        # Calculate min, max, and median BIC and AIC scores across repeats and initialisations for each Gaussian component count ie from n_init * n_repeats values - (n_gauss_components)
        BIC_min, BIC_max, BIC_median = BIC_scores.min(axis=(1, 2)), BIC_scores.max(axis=(1, 2)), np.nanmedian(BIC_scores, axis=(1, 2))
        AIC_min, AIC_max, AIC_median = AIC_scores.min(axis=(1, 2)), AIC_scores.max(axis=(1, 2)), np.nanmedian(AIC_scores, axis=(1, 2))

        # Plot combined BIC & AIC
        fig, ax = plt.subplots(figsize=(10, 6))
        # BIC - Blue
        ax.plot(n_gauss_list, BIC_min, 'b-', label="BIC - Lowest (Solid)")
        ax.plot(n_gauss_list, BIC_max, 'b--', label="BIC - Highest (Dashed)")
        ax.plot(n_gauss_list, BIC_median, 'b:', label="BIC - Median (Dotted)")

        # AIC - Red
        ax.plot(n_gauss_list, AIC_min, 'r-', label="AIC - Lowest (Solid)")
        ax.plot(n_gauss_list, AIC_max, 'r--', label="AIC - Highest (Dashed)")
        ax.plot(n_gauss_list, AIC_median, 'r:', label="AIC - Median (Dotted)")

        ax.set_xlabel("Number of Gaussian Components", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("BIC and AIC Score Analysis for Gaussian Components", fontsize=14)
        ax.legend(loc='best')
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        return None
    
    def assigment_XD(self, assigment_metric = 'best'): 
        """
        Assign stars to Gaussian components based on the best-fit XD model.
        Computes the responsibility of each gaussians for each star and assigns it accordingly.

        Parameters
        ----------
        assignment_metric : str
            Selection criteria for the best-fit model ('best' or 'best filtered').

        Raises
        ------
        ValueError
            If no XD results are available.
        ValueError
            If an invalid `assignment_metric` is specified.

        Returns
        -------
        None
            Updates `star_data` in place to include probability assignments:
                - `prob_gauss_{i}`: Probability of belonging to the i-th Gaussian component.
                - `max_gauss`: Index of the component with the highest probability (1-based index).
        """


        if self.results_XD is None:
            raise ValueError("No XD results found in class. Please run XD and analysis first or provide valid path to load from in the analysis method.")
        
        if assigment_metric not in ['best', 'best filtered']:
            raise ValueError("Invalid assignment metric selected. Please select either 'best' or 'best filtered'.")
        
        if assigment_metric == 'filtered' and self.filtered_best_params is None:
            raise ValueError("No filtered best parameters found. Please run filtered analysis first.")
        
        if assigment_metric == 'best' and self.best_params == {}:
            raise ValueError("No best parameters found. Please run analysis first.")
        
        elif self.best_BIC_score is None:
            raise ValueError("No best BIC score found. Please run BIC analysis first.")
        
        # Delete any columns in self.star_data that are already present from past analysis
        # Search for keys
        for key in self.star_data.keys():
            if key[:11] == 'prob_gauss_':
                del self.star_data[key]
        if 'max_gauss' in self.star_data.keys():
            del self.star_data['max_gauss']
        
        # Extract the results of relevent analysis locally
        if assigment_metric == 'filtered':
            assigment_params = self.filtered_best_params
        if assigment_metric == 'best filtered':
            assigment_params = self.best_params

        # Print a summary of what this is preforming
        print(f"Assigning stars to Gaussian components based on the {assigment_metric} XD model.")
        print(f"This has been optimised for the {assigment_params['metric']} score and returned the results:")
        print(f" Best {assigment_params['metric']} Score: {assigment_params['score']:.4f} occurred at:")
        print(f"   - Gaussian Components (n_gauss): {assigment_params['gauss_components']}")
        print(f"   - Repeat cycle (n): {assigment_params['repeat']}")
        print(f"   - Initialisation (i): {assigment_params['init']}")

        # Error-Aware Explanation:
        # Cannot evaluate the probability density at position in parameter space deirectly
        # XD accounts for measurement errors by modifying the covariance matrices of the Gaussian components. T
        # Allows total uncertainty reflects both the model and the measurement noise.
        # Done by adding the measurement error covariance X to the intrinsic Gaussian variance V, so:
        #     T  = V + Xerr
        
        # Initialises columns for probabilities and assignments
        for i in range(assigment_params['gauss_components']):
            self.star_data[f'prob_gauss_{i+1}'] = np.zeros(len(self.star_data))
            self.star_data['max_gauss'] = np.zeros(len(self.star_data), dtype=int)

        # For each star calculate the probability of it belonging to each gaussian
        for star_index, star in enumerate(self.feature_data):
            probabilities = []
            # Extract the measurement error covariance for the current star and conver it to a diagonal matrix
            star_errors = np.diag(self.errors_data[star_index])
            # Cycle through each of the gaussinas from the n_gauss
            for j in range(assigment_params['gauss_components']):
                # Mean covariance and weight for the jth gaussian
                mean_j = self.best_XD_means[j]
                # Error-Aware Covariance Adjustment
                cov_j = self.best_XD_covariances[j] + star_errors
                weight_j = self.best_XD_weights[j]

                # Calculate the probability of the data point given the gaussian
                try:
                    prob = weight_j * multivariate_normal.pdf(star, mean=mean_j, cov=cov_j)
                # Handle singular covariance matrices
                except np.linalg.LinAlgError:
                    prob = 0 

                probabilities.append(prob)
                self.star_data[f'prob_gauss_{j+1}'][star_index] = prob
            # Assign the star to the gaussian with the highest probability
            self.star_data['max_gauss'][star_index] = np.argmax(probabilities) + 1

        return None

    def plotting_XD(self):