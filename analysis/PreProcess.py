import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from astropy.io import fits
from astropy.table import Table
from astropy.table import vstack
from astroquery.gaia import Gaia

from scipy.stats import norm

import logging
logging.basicConfig(level=logging.INFO)

def convert_to_astropy_table(data):
    """
    Convert input data to an Astropy Table.
    
    Parameters
    ----------
    data : str, np.recarray, pd.DataFrame, or Table
        Input data, which can be a file path (CSV, FITS, TXT), a NumPy recarray, 
        a Pandas DataFrame, or already an Astropy Table.

    Returns
    -------
    Table
        Converted Astropy Table.

    Raises
    ------
    ValueError
        If the file type is unsupported or cannot be read.
    TypeError
        If the input data type is unsupported.
    """

    # If already an Astropy Table, return as-is
    if isinstance(data, Table):
        return data

    # Convert NumPy recarray to Astropy Table
    elif isinstance(data, np.recarray):
        return Table(data)

    # Convert Pandas DataFrame to Astropy Table
    elif isinstance(data, pd.DataFrame):
        return Table.from_pandas(data)

    # If input is a string, check if it is a file path and try to read it
    elif isinstance(data, str):
        # Extract file extension safely
        file_ext = os.path.splitext(data.lower())[1]

        try:
            if file_ext == ".csv":
                return Table.read(data, format="csv")
            elif file_ext in [".fits", ".fit"]:
                return Table.read(data, format="fits")
            elif file_ext in [".txt", ".dat"]:
                return Table.read(data, format="ascii")
            else:
                raise ValueError(f"Unsupported file type: {file_ext}. Supported types: CSV, FITS, TXT.")
        except Exception as e:
            raise ValueError(f"Failed to read file {data}: {e}")

    # If input type is unsupported, raise an error
    else:
        raise TypeError("Unsupported data type. Must be an Astropy Table, NumPy recarray, Pandas DataFrame, or a valid file path.")


def gallah_filter(star_data_in, dynamics_data_in, gaia_data_in, save_path=None):
    """
    Applies quality cuts to GALAH, Gaia, and dynamics datasets to produce 
    a refined sample of metal-poor, high-eccentricity stars. 

    This function filters stars based on data quality, chemical abundances, 
    orbital properties, and distance uncertainties.

    Parameters
    ----------
    star_data_in : str, Table, np.recarray, or pd.DataFrame
        GALAH stellar data, provided as a file path (CSV, FITS, TXT) or an 
        Astropy Table, NumPy recarray, or Pandas DataFrame.

    dynamics_data_in : str, Table, np.recarray, or pd.DataFrame
        Dynamics dataset containing orbital properties (e.g., energy, 
        eccentricity, actions).

    gaia_data_in : str, Table, np.recarray, or pd.DataFrame
        Gaia dataset providing distances and photogeometric uncertainties. 

    save_path : str, optional
        If provided, saves the filtered dataset as a FITS file at the specified path.

    Returns
    -------
    Table
        An Astropy Table containing the filtered stellar sample.

    Filtering Criteria
    ------------------
    **1. Data Quality Cuts**
       - `flag_sp == 0` → Only include stars with reliable stellar parameters.
       - `snr_c3_iraf > 30` → Ensure good signal-to-noise ratio (SNR).
       - `logg < 3.0` → Select only giant stars.

    **2. Element Abundance Filters**
       - `[Fe/H]`: Only stars with `flag_fe_h == 0` and `e_fe_h < 0.2`.
       - `[alpha/Fe]`: Only stars with `flag_alpha_fe == 0` and `e_alpha_fe < 0.2`.
       - `[Na/Fe]`: Remove unreliable Na measurements (`flag_Na_fe == 0` and `e_Na_fe < 0.2`).
       - `[Al/Fe]`: Remove unreliable Al measurements (`flag_Al_fe == 0` and `e_Al_fe < 0.2`).
       - `[Mn/Fe]`: Remove unreliable Mn measurements (`flag_Mn_fe == 0` and `e_Mn_fe < 0.2`).
       - `[Y/Fe]`: Remove unreliable Y measurements (`flag_Y_fe == 0` and `e_Y_fe < 0.2`).
       - `[Ba/Fe]`: Remove unreliable Ba measurements (`flag_Ba_fe == 0` and `e_Ba_fe < 0.2`).
       - `[Eu/Fe]`: Remove unreliable Eu measurements (`flag_Eu_fe == 0` and `e_Eu_fe < 0.2`).

    **3. Derived Element Ratio Filters**
       - `[Mg/Cu]`: Exclude stars with unreliable values (`flag_Mg_Cu == 0`, `e_Mg_Cu < 0.2`).
       - `[Mg/Mn]`: Exclude stars with unreliable values (`flag_Mg_Mn == 0`, `e_Mg_Mn < 0.2`).
       - `[Ba/Eu]`: Exclude stars with unreliable values (`flag_Ba_Eu == 0`, `e_Ba_Eu < 0.2`).

    **4. Orbital and Kinematic Cuts**
       - `Eccentricity > 0.85` → Select stars on highly radial orbits.
       - `Energy < 0` → Remove stars with unbound or positive energy.
       - `R_ap > 5` → Require an apocenter larger than 5 kpc to focus on outer halo structures.

    **5. Distance Uncertainty Cut (Gaia)**
       - `(r_hi_photogeo - r_med_photogeo) < 1500 pc` → Reject stars with large upper uncertainty.
       - `(r_med_photogeo - r_lo_photogeo) < 1500 pc` → Reject stars with large lower uncertainty.

    **6. Ensuring Data Consistency**
       - The GALAH, Gaia, and dynamics datasets are matched using `sobject_id`.
       - Datasets are ordered to maintain consistency.
       - Duplicate entries are removed.

    Output
    ------
    - The filtered dataset is returned as an Astropy Table.
    - If `save_path` is specified, the dataset is saved as a FITS file.

    Notes
    -----
    - These cuts aim to select metal-poor stars on extreme orbits, relevant 
      for studies of the Galactic halo and accretion history.
    - Stars that pass the filters will have **high-quality chemical abundances, 
      well-measured kinematics, and accurate distances**.
    """

    # Ensure that the input data can either be converted to an Astropy Table or is already an Astropy Table
    star_data = convert_to_astropy_table(star_data_in)
    dynamics_data = convert_to_astropy_table(dynamics_data_in)
    gaia_data = convert_to_astropy_table(gaia_data_in)

    # Store initial number of stars
    initial_star_count = len(star_data)
    logging.info(f"Initial number of stars: {initial_star_count}")

    # ------------------ REQUIRED KEYS CHECK ------------------

    # Define the required columns for each dataset
    required_keys = {
        "star_data": [
            "sobject_id", "flag_sp", "snr_c3_iraf", "logg", "flag_fe_h", "e_fe_h", "flag_alpha_fe", 
            "e_alpha_fe", "flag_Na_fe", "e_Na_fe", "flag_Al_fe", "e_Al_fe", "flag_Mn_fe", "e_Mn_fe",
            "flag_Y_fe", "e_Y_fe", "flag_Ba_fe", "e_Ba_fe", "flag_Eu_fe", "e_Eu_fe", "flag_Mg_fe",
            "e_Mg_fe", "flag_Cu_fe", "e_Cu_fe", "Mg_fe", "Cu_fe", "Mn_fe", "Ba_fe", "Eu_fe"
        ],
        "dynamics_data": ["sobject_id", "Energy", "Energy_5", "Energy_95", "ecc", "R_ap", "J_R", "L_Z", "J_Z"],
        "gaia_data": ["sobject_id", "r_med_photogeo", "r_lo_photogeo", "r_hi_photogeo"]
    }

    # Function to check missing keys in a dataset
    def check_missing_keys(dataset, dataset_name):
        missing_keys = [key for key in required_keys[dataset_name] if key not in dataset.colnames]
        if missing_keys:
            raise ValueError(f"Missing required columns in {dataset_name}: {missing_keys}")

    # Check all datasets for required keys
    check_missing_keys(star_data, "star_data")
    check_missing_keys(dynamics_data, "dynamics_data")
    check_missing_keys(gaia_data, "gaia_data")

    # ------------------ Begin Chemical Filtering (Gallah) ------------------
    # 1. Recommended Stellar Parameters Filter
    sp_filter = star_data['flag_sp'] == 0

    # 3. Recommended Signal to Noise Ratio Filter
    snr_filter = star_data['snr_c3_iraf'] > 30

    # 2. Filtering out bad star data
    rg_filter = star_data['logg'] < 3.0


    # 3. Element Abundance Filters
    # [Fe/H], [α/Fe], [Na/ Fe], [Al/Fe], [Mn/Fe], [Y/Fe], [Ba/Fe], [Eu/Fe], [Mg/Cu], [Mg/Mn], [Ba/Eu]
    # Fe/H filter
    fe_h_flag_filter = star_data['flag_fe_h'] == 0
    fe_h_err_filter = star_data['e_fe_h'] < 0.2
    fe_h_filter = fe_h_flag_filter & fe_h_err_filter


    # [α/Fe] filter
    alpha_fe_flag_filter = star_data['flag_alpha_fe'] == 0
    alpha_fe_err_filter = star_data['e_alpha_fe'] < 0.2
    alpha_fe_filter = alpha_fe_flag_filter & alpha_fe_err_filter

    # [Na/Fe] filter
    na_fe_flag_filter = star_data['flag_Na_fe'] == 0
    na_fe_err_filter = star_data['e_Na_fe'] < 0.2
    na_fe_filter = na_fe_flag_filter & na_fe_err_filter

    # [Al/Fe] filter
    al_fe_flag_filter = star_data['flag_Al_fe'] == 0
    al_fe_err_filter = star_data['e_Al_fe'] < 0.2
    al_fe_filter = al_fe_flag_filter & al_fe_err_filter

    # [Mn/Fe] filter
    mn_fe_flag_filter = star_data['flag_Mn_fe'] == 0
    mn_fe_err_filter = star_data['e_Mn_fe'] < 0.2
    mn_fe_filter = mn_fe_flag_filter & mn_fe_err_filter

    # [Y/Fe] filter
    y_fe_flag_filter = star_data['flag_Y_fe'] == 0
    y_fe_err_filter = star_data['e_Y_fe'] < 0.2
    y_fe_filter = y_fe_flag_filter & y_fe_err_filter

    # [Ba/Fe] filter
    ba_fe_flag_filter = star_data['flag_Ba_fe'] == 0
    ba_fe_err_filter = star_data['e_Ba_fe'] < 0.2
    ba_fe_filter = ba_fe_flag_filter & ba_fe_err_filter

    # [Eu/Fe] filter
    eu_fe_flag_filter = star_data['flag_Eu_fe'] == 0
    eu_fe_err_filter = star_data['e_Eu_fe'] < 0.2
    eu_fe_filter = eu_fe_flag_filter & eu_fe_err_filter

    # [Mg/Cu] filter
    if 'Mg_CU' not in star_data.colnames:
        star_data['Mg_Cu'] = star_data['Mg_fe'] - star_data['Cu_fe']

    if 'e_Mg_Cu' not in star_data.colnames:
        star_data['e_Mg_Cu'] = np.sqrt(star_data['e_Mg_fe']**2 + star_data['e_Cu_fe']**2)

    if 'flag_Mg_Cu' not in star_data.colnames:
        mg_fe_flag_filter = star_data['flag_Mg_fe'] == 0
        cu_fe_flag_filter = star_data['flag_Cu_fe'] == 0
        mg_cu_flag_filter = mg_fe_flag_filter & cu_fe_flag_filter
    else: 
        mg_cu_flag_filter = star_data['flag_Mg_Cu'] == 0

    mg_cu_err_filter = star_data['e_Mg_Cu'] < 0.2
    mg_cu_filter = mg_cu_flag_filter & mg_cu_err_filter

    # [Mg/Mn] filter
    if 'Mg_Mn' not in star_data.colnames:
        star_data['Mg_Mn'] = star_data['Mg_fe'] - star_data['Mn_fe']
    
    if 'e_Mg_Mn' not in star_data.colnames:
        star_data['e_Mg_Mn'] = np.sqrt(star_data['e_Mg_fe']**2 + star_data['e_Mn_fe']**2)
    
    if 'flag_Mg_Mn' not in star_data.colnames:
        mg_fe_flag_filter = star_data['flag_Mg_fe'] == 0
        mn_fe_flag_filter = star_data['flag_Mn_fe'] == 0
        mg_mn_flag_filter = mg_fe_flag_filter & mn_fe_flag_filter
    else:
        mg_mn_flag_filter = star_data['flag_Mg_Mn'] == 0
    
    mg_mn_err_filter = star_data['e_Mg_Mn'] < 0.2
    mg_mn_filter = mg_mn_flag_filter & mg_mn_err_filter

    # [Ba/Eu] filter
    if 'Ba_Eu' not in star_data.colnames:
        star_data['Ba_Eu'] = star_data['Ba_fe'] - star_data['Eu_fe']

    if 'e_Ba_Eu' not in star_data.colnames:
        star_data['e_Ba_Eu'] = np.sqrt(star_data['e_Ba_fe']**2 + star_data['e_Eu_fe']**2)

    if 'flag_Ba_Eu' not in star_data.colnames:
        ba_fe_flag_filter = star_data['flag_Ba_fe'] == 0
        eu_fe_flag_filter = star_data['flag_Eu_fe'] == 0
        ba_eu_flag_filter = ba_fe_flag_filter & eu_fe_flag_filter
    else:
        ba_eu_flag_filter = star_data['flag_Ba_Eu'] == 0

    ba_eu_err_filter = star_data['e_Ba_Eu'] < 0.2
    ba_eu_filter = ba_eu_flag_filter & ba_eu_err_filter

    # ------------------ Apply stage 1 filters ------------------
    star_data = star_data[sp_filter & snr_filter & rg_filter & fe_h_filter & alpha_fe_filter & 
                                   na_fe_filter & al_fe_filter & mn_fe_filter & y_fe_filter & ba_fe_filter & 
                                   eu_fe_filter & mg_cu_filter & mg_mn_filter & ba_eu_filter]
    
    # ------------------ Process tables so they can be combined ------------------
    # Order remaining stars by object ID
    star_data = star_data[np.argsort(star_data['sobject_id'])]

    # Filter Dynamics and Import data to enties match the star data
    dynamics_filter = np.isin(dynamics_data['sobject_id'], star_data['sobject_id'])
    gaia_filter = np.isin(gaia_data['sobject_id'], star_data['sobject_id'])

    dynamics_data = dynamics_data[dynamics_filter]
    gaia_data = gaia_data[gaia_filter]

    # Order them by object ID
    dynamics_data = dynamics_data[np.argsort(dynamics_data['sobject_id'])]
    gaia_data = gaia_data[np.argsort(gaia_data['sobject_id'])]

    # Assert that tables match and no duplicates
    if len(star_data) != len(dynamics_data) or len(star_data) != len(gaia_data):
        raise ValueError("Mismatch in number of rows between filtered star data, dynamics data, and Gaia data.")

    if not np.array_equal(star_data['sobject_id'], dynamics_data['sobject_id']) or not np.array_equal(star_data['sobject_id'], gaia_data['sobject_id']):
        raise ValueError("sobject_id mismatch between datasets. Ensure they have the same order and unique IDs.")

    # ------------------ Add dynamics data to central Gallah table ------------------
    # Energy
    star_data['Energy'] = dynamics_data['Energy']
    # Assume the energy error is a normal distribution
    # Tranform from 5, 95th percentile to standard deviation
    star_data['e_Energy'] = (dynamics_data['Energy_95'] - dynamics_data['Energy_5'])/ (norm.ppf(0.95) - norm.ppf(0.05))

    # Eccentricity
    star_data['Eccen'] = dynamics_data['ecc']

    # Apocenter
    star_data['R_ap'] = dynamics_data['R_ap']

    # Action variables
    star_data['J_R'] = dynamics_data['J_R']
    star_data['L_Z'] = dynamics_data['L_Z']
    star_data['J_Z'] = dynamics_data['J_Z']

    # ------------------ Add Gaia data to central Gallah table ------------------
    # Use photo_geometric distance rather than just photometric distance as offers more accuracy
    star_data['r_med_photogeo'] = gaia_data['r_med_photogeo']
    star_data['r_lo_photogeo'] = gaia_data['r_lo_photogeo']
    star_data['r_hi_photogeo'] = gaia_data['r_hi_photogeo']


    # ------------------ Filter Eccentricity and Energy and Apocenter ------------------
    ecc_filter = star_data['Eccen'] > 0.85
    energy_filter = star_data['Energy'] < 0
    apocenter_filter = star_data['R_ap'] > 5

    # ------------------ Filter for distance uncert ------------------
    # Uncertainty less than 1.5 kpc - both upper and lower bounds taken seperately to be rigorous
    dist_err_filter_hi = (star_data['r_hi_photogeo']-star_data['r_med_photogeo']) < 1500
    dist_err_filter_lo = (star_data['r_med_photogeo']-star_data['r_lo_photogeo']) < 1500

    # ------------------ Apply stage 2 filters ------------------
    star_data = star_data[ecc_filter & energy_filter & apocenter_filter &
                                                dist_err_filter_hi & dist_err_filter_lo]
    
    # Store final number of stars
    final_star_count = len(star_data)
    logging.info(f"Final number of stars: {final_star_count}")
    logging.info(f"Fraction retained: {final_star_count / initial_star_count:.2%}")
    
    # ------------------ Save filtered data if path provided ------------------

    # Save data if a path is provided
    if save_path:
        star_data.write(save_path, format="fits", overwrite=True)
        logging.info(f"Filtered dataset saved to {save_path}")

    
    return star_data   



def apogee_filter(data):


    return None