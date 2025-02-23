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
        Initialise the XDPipeline with the star data and relevant keys/ features that the Gaussian Mixture Model (XD) will be fitted to. Defining the parameter space of interest. 

        Parameters:
            star_data (Table, np.recarray, pd.DataFrame): The dataset containing star information.
            data_keys (List[str]): List of keys for the parameters of interest in the data table.
            data_err_keys (List[str]): List of keys for the errors of the parameters of interest in the data table, in order corresponding to the data_keys.

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

        # The best BIC score and the parameters in which it was achieved
        self.best_BIC_score = None
        self.best_BIC_index = None
        self.best_repeat = None
        self.best_init = None
        self.best_gauss_components = None
        self.best_XD_weights = None
        self.best_XD_means = None
        self.best_XD_covariances = None



    def _BICScore(self, log_likelihood: float, num_params: int, num_data_points: int) -> float:
            """
            Bayesian Information Criterion (BIC) 

            A model selection criterion that balances model fit and complexity by penalizing the number of parameters more strongly than AIC, especially for larger datasets.

            Parameters: 
                log_likelihood (float): Overall Log likelihood of the model
                num_params (int): Number of parameters in the model
                num_data_points (int): Number of data points in the dataset

            Returns:
                float: BIC score
            """
            return -2 * log_likelihood + num_params * np.log(num_data_points)
    
    def _AICScore(self, log_likelihood: float, num_params: int) -> float:
        """
        Akaike Information Criterion (AIC)

        A model selection criterion that balances model fit and complexity by penalizing the number of parameters with a constant factor (2 per parameter).

        Parameters: 
            log_likelihood (float): Overall Log likelihood of the model
            num_params (int): Number of parameters in the model

        Returns:
            float: AIC score
        """
        return - 2 * log_likelihood + 2 * num_params
    
    def run_XD(self, gauss_component_range: Tuple[int, int] = (1, 10), max_iterations: int = int(1e10), n_repeats: int = 3, n_init: int = 100, save_path: Optional[str] = None) -> None:
        """
        Run Extreme Deconvolution with varying number of Gaussian components and multiple initialisations to ensure convergence - allowing for model selection using BIC and AIC scores.

        Parameters:
            component_range (tuple): Range of gaussian components to test, (min, max).
            max_iterations (int): Maximum number of EM iterations for each run.
            n_repeats (int): Number of complete repetitions of the 100 initialisations.
            n_init (int): Number of random initializations per component count.
            save_path (str): Path to save the results to if desired.

        Returns:
            None
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
    
    def BIC_analysis(self, save_path: Optional[str] = None) -> None:
        """
        Analyse Extreme Deconvolution (XD) results using BIC and AIC criteria.

        This method identifies the best-fitting Gaussian mixture model based on the lowest BIC score, 
        summarizes failed XD runs, computes mean and standard deviation of BIC and AIC scores, 
        and visualizes these metrics across Gaussian component counts. Results can be loaded 
        from a pickle file if not already available.

        Parameters
        ----------
        save_path : Optional[str], default=None
            Path to a pickle file with XD results.  Used if no existing results in the class.

        Returns
        -------
        None
            Outputs summary tables, updates best-fit model attributes, and displays BIC/AIC plots.

        Raises
        ------
        ValueError
            If no results are found in the class and no valid `save_path` is provided.
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
            self.n_init = max(self.results_XD["init no"]) + 1


        # Identifies the best BIC score - its index, correlated inputs and results
        self.best_BIC_score = min(b for b in self.results_XD["BIC"] if b is not None)
        self.best_BIC_index = self.results_XD["BIC"].index(self.best_BIC_score)
        self.best_gauss_components = self.results_XD["n_gauss"][self.best_BIC_index]
        self.best_repeat = self.results_XD["repeat no."][self.best_BIC_index]
        self.best_init = self.results_XD["init no"][self.best_BIC_index]
        self.best_XD_weights = self.results_XD["weights"][self.best_BIC_index]
        self.best_XD_means = self.results_XD["means"][self.best_BIC_index]
        self.best_XD_covariances = self.results_XD["covariances"][self.best_BIC_index]

        # Prints summary of best performing Gaussian components
        print(f" Best BIC Score: {self.best_BIC_score:.4f} occurred at:")
        print(f"   - Gaussian Components (n_gauss): {self.best_gauss_components}")
        print(f"   - Repeat cycle (n): {self.best_repeat}")
        print(f"   - Initialisation (i): {self.best_init}")

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
    
    def XD_assigment(self): 
        """
        Assign stars to Gaussian components based on the best-fit XD model.

        Calculates the probability of each star belonging to each Gaussian component from 
        the best BIC-scoring XD model and assigns stars to the component with the highest probability. 
        Adds the following columns to `star_data`:
        
        - `prob_gauss_{i}`: Probability of belonging to the i-th Gaussian component.
        - `max_gauss`: Index of the component with the highest probability (1-based).

        Raises
        ------
        ValueError
            If XD results (`self.results_XD`) or BIC analysis (`self.best_BIC_score`) are missing.

        Returns
        -------
        None
            Updates `star_data` in place with probability assignments and component labels.
        """


        if self.results_XD is None:
            raise ValueError("No XD results found in class. Please run XD and analysis first or provide valid path to load from in the analysis method.")
        elif self.best_BIC_score is None:
            raise ValueError("No best BIC score found. Please run BIC analysis first.")
        
        # Initialises columns for probabilities and assignments
        for i in range(self.best_gauss_components):
            self.star_data[f'prob_gauss_{i+1}'] = np.zeros(len(self.star_data))
            self.star_data['max_gauss'] = np.zeros(len(self.star_data), dtype=int)

        # For each star calculate the probability of it belonging to each gaussian
        for star_index, star in enumerate(self.feature_data):
            probabilities = []
            # Cycle through each of the gaussinas from the n_gauss
            for j in range(self.best_gauss_components):
                # Mean covariance and weight for the jth gaussian
                mean_j = self.best_XD_means[j]
                cov_j = self.best_XD_covariances[j]
                weight_j = self.best_XD_weights[j]
                # Calculate the probability of the data point given the gaussian
                prob = weight_j * multivariate_normal.pdf(star, mean=mean_j, cov=cov_j)
                probabilities.append(prob)
                self.star_data[f'prob_gauss_{j+1}'][star_index] = prob
            # Assign the star to the gaussian with the highest probability
            self.star_data['max_gauss'][star_index] = np.argmax(probabilities) + 1

        return None





        





# # The issue i am facing is that it is only fitting one of the gaussians the rest just get their covariances matrix set to almost zero. 
# def run_XD_BIC(data, data_keys: List[str], data_err_keys: List[str], component_range: Tuple[int, int] = (1, 10), max_iterations: int = int(1e10), n_repeats: int = 3, n_init: int = 100, save_path: Optional[str] = None) -> Tuple[dict, dict]:
#     """
#     Run Extreme Deconvolution with varying number of Gaussian components and multiple initialisations to ensure convergence allowing for model selection using BIC and AIC scores.

#     Parameters:
#         data (Astropy Table): Full dataset, including parameters and their errors with additional columns not used for XD.
#         data_keys (list): List of keys for the parameters of interest in the data table.
#         data_err_keys (list): List of keys for the errors of the parameters of interest in the data table. These must be in order and correspond to the data_keys.
#         component_range (tuple): Range of gaussian components to test, (min, max).
#         max_iterations (int): Maximum number of EM iterations for each run.
#         n_repeats (int): Number of complete repetitions of the 100 initialisations.
#         n_init (int): Number of random initializations per component count.
#         save_path (str): Path to save the results to.

#     Returns:
#         dict: Contains BIC, AIC, and best-fit parameters for each component count.
#         dict: Contains the best BIC score and the parameters in which it was achieved
#     """

#     # Entry checks
#     # Check that the number of data keys and error keys match
#     if len(data_keys) != len(data_err_keys):
#         raise ValueError("Number of data keys and error keys must match")

#     # Check if data is an Astropy Table or recarray and extract column names accordingly
#     if hasattr(data, 'colnames'):  # For Astropy Table
#         colnames = data.colnames
#     elif hasattr(data, 'dtype'):   # For recarray (FITS_rec)
#         colnames = data.dtype.names
#     else:
#         raise TypeError("Unsupported data type. Must be Astropy Table or FITS_rec/recarray.")
    
#     # Check that the data keys and error keys are present in the data table
#     # Print the missing keys if they are not present
#     missing_keys = [key for key in data_keys + data_err_keys if key not in colnames]
#     if missing_keys:
#         raise ValueError(f"Keys {missing_keys} not found in data table")
    
#     # Check that the component range is valid
#     if not isinstance(component_range, tuple) or len(component_range) != 2:
#         raise ValueError("Gaussian component range must be a tuple of form (min, max)")
    
#     if component_range[0] > component_range[1]:
#         raise ValueError("Invalid gaussian component range") 
    
#     # Check that the number of repeats and initialisations are valid
#     if not isinstance(n_repeats, int) or n_repeats <= 0:
#         raise TypeError("n_repeats must be a positive integer")
#     if not isinstance(n_init, int) or n_init <= 0:
#         raise TypeError("n_init must be a positive integer")
#     if not isinstance(max_iterations, int) or max_iterations <= 0:
#         raise TypeError("max_iterations must be a positive integer")



#     # Extract the number of samples and features from the data array
#     # Alternatively, n_features = len(data_keys) and n_samples = len(data_array)
#     n_samples, n_features = data_array.shape

#     # Scale the data to have zero mean and unit variance - this will improve the convergence of the EM algorithm
#     # Errors are scaled by the same factor to maintain the same relative uncertainty
#     # Note this will require the gaussians means and covariances returned by XD to be scaled back to the original units for interpretation before saving
#     scaler = StandardScaler()
#     data_scaled = scaler.fit_transform(data_array)
#     errors_scaled = errors_array / scaler.scale_

#     # Calculate the extreme values of the data for initialisation randomisation
#     extreme_data_values = (np.max(data_array, axis=0), np.min(data_array, axis=0))

#     # Initialise the results dictionary for dynamic appending
#     results = {
#         # Repeat Number
#         "repeat no.": [],
#         # Intialisation Number
#         "init no": [],
#         # Number of gaussians fitted
#         "n_gauss": [],
#         # Log likelihood of the best-fit model
#         "log_likelihood": [],
#         # Bayesian Information Criterion
#         "BIC": [],
#         # Akaike Information Criterion
#         "AIC": [],
#         # Best-fit weights of the gaussians
#         "weights": [],
#         # Best-fit means of the gaussians
#         "means": [],
#         # Best-fit covariances of the gaussians
#         "covariances": []
#     }

#     # Track the best BIC and initialise the best parameters
#     best_BIC = np.inf
#     best_params = {"BIC": None, "n_gauss": None, "repeat": None, "init": None, "means": None}

#     # Iterate for a test range of number of gaussians
#     for n_gauss in tqdm(range(component_range[0], component_range[1] + 1), desc="Number of Gaussian Components"):
#         # Overall repeats 
#         for n in tqdm(range(n_repeats), desc="Repeat Cycles", leave=False):
#             # Random initialisations of input parameters
#             for i in tqdm(range(n_init), desc="Initialisations", leave=False):

#                 # Random initiatlisation of weights
#                 # init_weights = np.random.dirichlet(np.ones(n_gauss))
#                 # Even initialisation of weights
#                 init_weights = np.ones(n_gauss) / n_gauss

#                 # Random initialisation of means - using the extreme values of each parameter
#                 init_mean = np.random.uniform(low=extreme_data_values[0], high=extreme_data_values[1], size=(n_gauss, n_features))

#                 # Covariances initialised as identity matrices
#                 init_covar = np.array([np.identity(n_features) for _ in range(n_gauss)])
#                 # Run XD
#                 try:
#                     XD_avg_LL = extreme_deconvolution(
#                         data_scaled, errors_scaled, init_weights, init_mean, init_covar, maxiter=max_iterations)
#                     # Calculate the total log likelihood
#                     total_LL = XD_avg_LL * n_samples

#                     # Calculate the bic and aic scores
#                     num_params = n_gauss * (1 + n_features + n_features * (n_features + 1) // 2) - 1
#                     bic, aic = BICScore(total_LL, num_params, n_samples), AICScore(total_LL, num_params)

#                     # Copy the updated weights, means and covariances
#                     post_XD_weights, post_XD_means, post_XD_cov = init_weights.copy(), init_mean.copy(), init_covar.copy()

#                     # Unscale the means and covariances to return them to their original/meaningful units
#                     post_scaling_means = scaler.inverse_transform(post_XD_means)
#                     post_scaling_cov = np.array([
#                         np.dot(np.dot(np.diag(scaler.scale_), cov), np.diag(scaler.scale_))
#                         for cov in post_XD_cov
#                     ])

#                     # Keep track of the best BIC and parameters in which it was achieved
#                     if bic < best_BIC:
#                         best_BIC = bic
#                         best_params.update({"BIC": bic, "n_gauss": n_gauss, "repeat": n, "init": i, "means": post_scaling_means})

#                     # Store the results
#                     results["repeat no."].append(n)
#                     results["init no"].append(i)
#                     results["n_gauss"].append(n_gauss)
#                     results["log_likelihood"].append(total_LL)
#                     results["BIC"].append(bic)
#                     results["AIC"].append(aic)
#                     results["weights"].append(post_XD_weights)
#                     results["means"].append(post_scaling_means)
#                     results["covariances"].append(post_scaling_cov)


#                 except Exception as e:
#                     print(f"XD failed for {n_gauss} components, on repeat: {n}, iteration: {i}: {e}")

#                     # Store the results
#                     results["repeat no."].append(n)
#                     results["init no"].append(i)
#                     results["n_gauss"].append(n_gauss)
#                     results["log_likelihood"].append(None)
#                     results["BIC"].append(None)
#                     results["AIC"].append(None)
#                     results["weights"].append(None)
#                     results["means"].append(None)
#                     results["covariances"].append(None)

#         # Save the results if a path is provided
#         # This is redone for each component count to ensure that the results are saved in case of a crash
#         if save_path:
#             try:
#                 with open(save_path, "wb") as f:
#                     pkl.dump(results, f)
#                 print(f"Results saved successfully at {save_path}")
#             except Exception as e:
#                 print(f"Failed to save results: {e}")

#     # Final summary of best BIC
#     print(f" Best BIC Score: {best_BIC:.4f}")
#     print(f"   - Gaussian Components (n_gauss): {best_params['n_gauss']}")
#     print(f"   - Repeat cycle (n): {best_params['repeat']}")
#     print(f"   - Initialisation (i): {best_params['init']}")
#     print(f"   - Best initialisation of Mean values: \n{best_params['means']}")

#     return results, best_params


# # Correction I want to be able to pass the dictionary in directly to the function or the path to the pickle file and it will load the dictionary - either way it will work out the format and if it needs to load from the path
# def BIC_analysis(saved_path):
#     """
#     Analyse the results of the Extreme Deconvolution (XD) 
    
#     Runs by computing and visualising the BIC/AIC scores for each Gaussian component count.

#     This function:
#     - Loads XD results from a pickle file.
#     - Calculates failed XD runs per Gaussian component.
#     - Computes mean and standard deviation of BIC and AIC scores.
#     - Displays formatted tables of results.
#     - Plots combined BIC and AIC curves with lowest, highest, and median scores.

#     Parameters:
#     ------------
#     saved_path : str
#         The file path to the pickle file containing XD results, including BIC and AIC scores.

#     Returns:
#     --------
#     None
#         Displays tables and plots summarizing the BIC/AIC analysis for each Gaussian component count.
#     """

#     # Loads the saved results from the XD run
#     with open(saved_path, "rb") as f:
#         XD_results = pkl.load(f)

#     # Extracts unique Gaussian component counts and define the range
#     component_range = (min(XD_results["n_gauss"]), max(XD_results["n_gauss"]))
#     n_gauss_list = np.array([n for n in range(component_range[0], component_range[1] + 1)])

#     # Determines the number of initializations and repeats
#     n_init = max(XD_results["init no"]) + 1
#     n_repeats = max(XD_results["repeat no."]) + 1

#     # Creates a DataFrame summarising the number of failed XD runs per Gaussian component
#     n_XD_failed = [
#         sum([b is None for c, b in zip(XD_results["n_gauss"], XD_results["BIC"]) if c == n_gauss])
#         for n_gauss in n_gauss_list
#     ]
#     n_runs_gauss = np.array([n_repeats * n_init for _ in n_gauss_list])
#     n_failed_XD = pd.DataFrame({
#         "No. Gaussians": n_gauss_list,
#         "No. Failed XD runs": n_XD_failed,
#         "Total No. Runs": n_runs_gauss
#     })

#     # Prints a formatted table of the number of failed XD runs
#     print("Table of Number of Gaussians vs Number of Failed XD Runs")
#     print(tabulate(n_failed_XD, headers='keys', tablefmt='psql'))

#     # Reshapes BIC and AIC scores to 3D arrays: (n_gauss_components, n_repeats, n_init)
#     BIC_scores = np.array(XD_results['BIC']).reshape((len(n_gauss_list), n_repeats, n_init))
#     AIC_scores = np.array(XD_results['AIC']).reshape((len(n_gauss_list), n_repeats, n_init))

#     # Compute means and standard deviations for BIC and AIC across initialisations
#     BIC_means, BIC_stds = np.mean(BIC_scores, axis=2), np.std(BIC_scores, axis=2)
#     AIC_means, AIC_stds = np.mean(AIC_scores, axis=2), np.std(AIC_scores, axis=2)

#     # Format mean ± stddev for BIC and AIC into DataFrames
#     BIC_means_stds_df = pd.DataFrame(
#         np.array([[f"{mean:.5f} \\pm {std:.5f}" for mean, std in zip(mean_row, std_row)]
#                   for mean_row, std_row in zip(BIC_means, BIC_stds)]),
#         columns=[f"Repeat {i + 1}" for i in range(n_repeats)],
#         index=n_gauss_list
#     )
#     BIC_means_stds_df.insert(0, "No. Gaussians", n_gauss_list)

#     AIC_means_stds_df = pd.DataFrame(
#         np.array([[f"{mean:.5f} \\pm {std:.5f}" for mean, std in zip(mean_row, std_row)]
#                   for mean_row, std_row in zip(AIC_means, AIC_stds)]),
#         columns=[f"Repeat {i + 1}" for i in range(n_repeats)],
#         index=n_gauss_list
#     )
#     AIC_means_stds_df.insert(0, "No. Gaussians", n_gauss_list)

#     # Print BIC and AIC summary tables
#     print("Table of BIC Means and Stds")
#     print(tabulate(BIC_means_stds_df, headers='keys', tablefmt='psql'))

#     print("Table of AIC Means and Stds")
#     print(tabulate(AIC_means_stds_df, headers='keys', tablefmt='psql'))

#     # Calculate min, max, and median BIC and AIC scores across repeats and initialisations for each Gaussian component count ie from n_init * n_repeats values - (n_gauss_components)
#     BIC_min, BIC_max, BIC_median = BIC_scores.min(axis=(1, 2)), BIC_scores.max(axis=(1, 2)), np.median(BIC_scores, axis=(1, 2))
#     AIC_min, AIC_max, AIC_median = AIC_scores.min(axis=(1, 2)), AIC_scores.max(axis=(1, 2)), np.median(AIC_scores, axis=(1, 2))

#     # Plot combined BIC & AIC
#     fig, ax = plt.subplots(figsize=(10, 6))
#     # BIC - Blue
#     ax.plot(n_gauss_list, BIC_min, 'b-', label="BIC - Lowest (Solid)")
#     ax.plot(n_gauss_list, BIC_max, 'b--', label="BIC - Highest (Dashed)")
#     ax.plot(n_gauss_list, BIC_median, 'b:', label="BIC - Median (Dotted)")

#     # AIC - Red
#     ax.plot(n_gauss_list, AIC_min, 'r-', label="AIC - Lowest (Solid)")
#     ax.plot(n_gauss_list, AIC_max, 'r--', label="AIC - Highest (Dashed)")
#     ax.plot(n_gauss_list, AIC_median, 'r:', label="AIC - Median (Dotted)")

#     ax.set_xlabel("Number of Gaussian Components", fontsize=12)
#     ax.set_ylabel("Score", fontsize=12)
#     ax.set_title("BIC and AIC Score Analysis for Gaussian Components", fontsize=14)
#     ax.legend(loc='best')
#     ax.grid(True)
#     plt.tight_layout()
#     plt.show()

#     return None

# # This function will take the original data set of stars and the results of the XD analysis 
# # It will determine the number of gaussians and the mean and covaraince matrices which achieve the best BIC score
# # It will then use the data set to assign each star a weighting to each gaussian component as well as labelling the star whith its maximium gaussian component. 
# # It will need to take the keys in order to extract the relevent data from th table
# # It will return the data table with the additional columns of the weights and the maximum gaussian component - note we do not know how many gaussian components and this columns will be needed
# def XD_assigment(star_data, BIC_data, data_keys): 


#     # Identifies the best BIC score - its index and correlated inputs
#     best_BIC = min(BIC_data["BIC"])
#     best_BIC_index = BIC_data["BIC"].index(best_BIC)
#     best_n_gauss = BIC_data["n_gauss"][best_BIC_index]
#     best_repeat = BIC_data["repeat no."][best_BIC_index]
#     best_init = BIC_data["init no"][best_BIC_index]

#     # Prints summary of best performing Gaussian components
#     print(f" Best BIC Score: {best_BIC:.4f} occurred at:")
#     print(f"   - Gaussian Components (n_gauss): {best_n_gauss}")
#     print(f"   - Repeat cycle (n): {best_repeat}")
#     print(f"   - Initialisation (i): {best_init}")

#     # Retrieves the number of gaussians, the mean vectors, and the covariance matrices of each gaussian
#     means = BIC_data["means"][best_BIC_index]
#     covariances = BIC_data["covariances"][best_BIC_index]
#     weights = BIC_data["weights"][best_BIC_index]

#     # Extracts the data from the astropy table using the keys provided and stack them into a 2D array, (number_of_parameters, number_of_samples)
#     star_data_array = np.vstack([star_data[key] for key in data_keys]).T

#     # Initialises columns for probabilities and assignments
#     for i in range(best_n_gauss):
#         star_data[f'prob_gauss_{i+1}'] = np.zeros(len(star_data))
#     star_data['max_gauss'] = np.zeros(len(star_data), dtype=int)

#     # For each star calculate the probability of it belonging to each gaussian
#     for star_index, star in enumerate(star_data_array):
#         probabilities = []
#         # Cycle through each of the gaussinas from the n_gauss
#         for j in range(best_n_gauss):
#             # Mean covariance and weight for the jth gaussian
#             mean_j = means[j]
#             cov_j = covariances[j]
#             weight_j = weights[j]
#             # Calculate the probability of the data point given the gaussian
#             prob = weight_j * multivariate_normal.pdf(star, mean=mean_j, cov=cov_j)
#             probabilities.append(prob)
#             star_data[f'prob_gauss_{j+1}'][star_index] = prob
#         # Assign the star to the gaussian with the highest probability
#         star_data['max_gauss'][star_index] = np.argmax(probabilities) + 1

#     return star_data

