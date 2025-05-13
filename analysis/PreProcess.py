import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from astropy.io import fits
from astropy.table import Table, vstack
from astroquery.gaia import Gaia
from astropy import units as u
from astropy.table import join

from scipy.stats import norm
import warnings
from astropy.units import UnitsWarning

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
                # Suppress unit warnings only when reading FITS
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UnitsWarning)
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


def galah_filter(star_data_in, dynamics_data_in, gaia_data_in, save_path=None):
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
    star_data['phot_g_mean_mag'] = gaia_data['phot_g_mean_mag']


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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UnitsWarning)
            star_data.write(save_path, format="fits", overwrite=True)
    logging.info(f"Filtered dataset saved to {save_path}")

    return star_data   



def apogee_filter(star_data_in, SQL=False, save_path=None):
    """
    Applies quality cuts to APOGEE stellar data to produce a refined 
    sample of chemically selected stars with extreme kinematics.

    This function filters stars based on data quality, chemical abundances, 
    and orbital properties to isolate **metal-poor stars with extreme orbits**.

    If `SQL=True`, Gaia DR3 distances are queried using the **astroquery** package, 
    and an additional filtering step removes stars with large distance errors.

    Parameters
    ----------
    star_data_in : str, Table, np.recarray, or pd.DataFrame
        APOGEE stellar data, provided as a file path (CSV, FITS, TXT) or an 
        Astropy Table, NumPy recarray, or Pandas DataFrame.

    SQL : bool, optional
        If `True`, queries Gaia DR3 for distances using `astroquery.gaia` 
        and applies additional filtering based on distance uncertainties.
        Defaults to `False`.

    save_path : str, optional
        If provided, saves the filtered dataset as a FITS file at the specified path.

    Returns
    -------
    Table
        An Astropy Table containing the filtered stellar sample.

    Filtering Criteria
    ------------------
    **1. Data Quality Cuts**
       - `extratarg == 0` → Select only Main Red Stars (MRS).
       - `logg < 3.0` → Restrict to giant stars.

    **2. Element Abundance Filters**
       - `[Fe/H]`: Require `fe_h_flag == 0` and `fe_h_err < 0.1` for reliable iron abundance.
       - `[Al/Fe]`: Require `al_fe_flag == 0` and `al_fe_err < 0.1` for accurate aluminum measurement.
       - `[Ce/Fe]`: Require `ce_fe_flag == 0` and `ce_fe_err < 0.15` for precise cerium abundance.

    **3. Derived Element Ratio Filters**
       - `[Mg/Mn]`: If missing, computed as `mg_fe - mn_fe`. Require:
         - `mg_mn_flag == 0` (or `mg_fe_flag == 0` & `mn_fe_flag == 0` if `mg_mn_flag` is missing).
         - `mg_mn_err < 0.1` for reliable measurement.
       - `[alpha/Fe]`: Constructed if missing from individual elements. Require:
         - `alpha_fe_flag == 0` and `alpha_fe_err < 0.1`.

    **4. Orbital and Kinematic Cuts**
       - `Eccentricity (ecc_50) > 0.85` → Select stars on highly radial orbits.
       - `Energy (E_50) < 0` → Remove unbound or high-energy stars.

    **5. Additional Filters (If SQL=True)**
       - Queries **Gaia DR3** for distances (`r_med_photogeo`, `r_lo_photogeo`, `r_hi_photogeo`).
       - **Distance Uncertainty Cut**: Rejects stars with:
         - `(r_hi_photogeo - r_med_photogeo) < 1500` pc (upper bound uncertainty)
         - `(r_med_photogeo - r_lo_photogeo) < 1500` pc (lower bound uncertainty)

    **6. Ensuring Data Consistency**
       - Checks for required keys before filtering.
       - Drops stars with missing values in `ecc_50` or `E_50`.
       - Orders dataset to maintain consistency.
       - Ensures Gaia ID (`GAIAEDR3_SOURCE_ID` or `dr3_source_id`) is present when `SQL=True`.

    Output
    ------
    - The filtered dataset is returned as an Astropy Table.
    - If `save_path` is specified, the dataset is saved as a FITS file.

    Notes
    -----
    - This selection aims to **isolate metal-poor stars with extreme orbits**, 
      relevant for Galactic archaeology and halo studies.
    - Stars that pass the filters have **high-quality chemical abundances, 
      well-measured kinematics, and a robust selection based on APOGEE data**.
    """
    # Ensure that the input data can either be converted to an Astropy Table or is already an Astropy Table
    star_data = convert_to_astropy_table(star_data_in)

    # Store initial number of stars
    initial_star_count = len(star_data)
    logging.info(f"Initial number of stars: {initial_star_count}")

    # ------------------ REQUIRED KEYS CHECK ------------------

    required_keys = [
        "extratarg", "logg", "fe_h", "fe_h_err", "fe_h_flag", "al_fe", "al_fe_err", "al_fe_flag",
        "ce_fe", "ce_fe_err", "ce_fe_flag", "mg_fe", "mg_fe_err", "mg_fe_flag",
        "mn_fe", "mn_fe_err", "mn_fe_flag", "alpha_fe_err", "ecc_50", "E_50"
    ]

    # If  SQL requirement is true, add the Gaia ID column - whether that be `GAIAEDR3_SOURCE_ID` or `dr3_source_id`
    if SQL:
        # Check if `GAIAEDR3_SOURCE_ID` or `dr3_source_id` exists
        if "GAIAEDR3_SOURCE_ID" in star_data.colnames:
            gaia_id_col = "GAIAEDR3_SOURCE_ID"
        elif "dr3_source_id" in star_data.colnames:
            gaia_id_col = "dr3_source_id"
        else:
            raise ValueError("SQL=True, but no Gaia source ID column (`GAIAEDR3_SOURCE_ID` or `dr3_source_id`) found in dataset.")
        
        # Add the correct Gaia ID column
        required_keys.append(gaia_id_col) 


    # Function to check missing keys
    missing_keys = [key for key in required_keys if key not in star_data.colnames]
    if missing_keys:
        raise ValueError(f"Missing required columns in star_data: {missing_keys}")


    # ------------------ Filtering Data ------------------

    # 1. Main Red Stars Filter
    mrs_filter = star_data['extratarg'] == 0

    # 2. Filtering out bad star data
    # bs_filter = star_data['ASPCAPFLAG'] != 'STAR_BAD'
    # prog_filter = star_data['PROGRAMNAME'] != 'magclouds'
    rg_filter = star_data['logg'] < 3.0

    # 3. Element Abundance Filters

    # Fe/H filter
    fe_h_flag_filter = star_data['fe_h_flag'] == 0
    # ASSUME THIS WAS SUPPOSED Top BE DONE
    fe_h_err_filter = star_data['fe_h_err'] < 0.1
    fe_h_filter = fe_h_flag_filter & fe_h_err_filter

    # Al/Fe filter
    al_fe_flag_filter = star_data['al_fe_flag'] == 0
    al_fe_err_filter = star_data['al_fe_err'] < 0.1
    al_fe_filter = al_fe_flag_filter & al_fe_err_filter
    # Ce/Fe filter
    ce_fe_flag_filter = star_data['ce_fe_flag'] == 0
    ce_fe_err_filter = star_data['ce_fe_err'] < 0.15
    ce_fe_filter = ce_fe_flag_filter & ce_fe_err_filter

    # Mg/Mn filter
    # Data Values
    if 'mg_mn' not in star_data.colnames:
        star_data['mg_mn'] = star_data['mg_fe'] - star_data['mn_fe']
    
    # Flag filter
    if 'mg_mn_flag' not in star_data.colnames:
        mg_fe_flag_filter = star_data['mg_fe_flag'] == 0
        mn_fe_flag_filter = star_data['mn_fe_flag'] == 0
        mg_mn_flag_filter = mg_fe_flag_filter & mn_fe_flag_filter
    else:
        mg_mn_flag_filter = star_data['mg_mn_flag'] == 0

    # Error values
    if 'mg_mn_err' not in star_data.colnames: 
        star_data['mg_mn_err'] = np.sqrt(star_data['mg_fe_err']**2 + star_data['mn_fe_err']**2)
    
    mg_mn_err_filter = star_data['mg_mn_err'] < 0.1
    mg_mn_filter = mg_mn_flag_filter & mg_mn_err_filter

    # Alpha/Fe filter
    if 'alpha_fe_flag' not in star_data.colnames:
        o_fe_flag_filter = star_data['o_fe_flag'] == 0
        mg_fe_flag_filter = star_data['mg_fe_flag'] == 0
        si_fe_flag_filter = star_data['si_fe_flag'] == 0
        ca_fe_flag_filter = star_data['ca_fe_flag'] == 0
        ti_fe_flag_filter = star_data['ti_fe_flag'] == 0
        alpha_fe_flag_filter = o_fe_flag_filter & mg_fe_flag_filter & si_fe_flag_filter & ca_fe_flag_filter & ti_fe_flag_filter
    else: 
        alpha_fe_flag_filter = star_data['alpha_fe_flag'] == 0

    # Update this
    if 'alpha_m_err' not in star_data.colnames:
        o_fe_error_filter = star_data['o_fe_err'] < 0.1
        mg_fe_error_filter = star_data['mg_fe_err'] < 0.1   
        si_fe_error_filter = star_data['si_fe_err'] < 0.1
        ca_fe_error_filter = star_data['ca_fe_err'] < 0.1
        ti_fe_error_filter = star_data['ti_fe_err'] < 0.1
        alpha_fe_err_filter = o_fe_error_filter & mg_fe_error_filter & si_fe_error_filter & ca_fe_error_filter & ti_fe_error_filter
    else:
        alpha_fe_err_filter = star_data['alpha_m_err'] < 0.1

    alpha_fe_filter = alpha_fe_flag_filter & alpha_fe_err_filter 

    # ------------------ Applying Stage 1 Filters ------------------
    # Extract only Main Red Stars
    apogee_data_red = star_data[mrs_filter]

    # Apply all filters to get the final cleaned dataset
    star_data = star_data[mrs_filter & rg_filter & fe_h_filter & al_fe_filter 
                            & ce_fe_filter & mg_mn_filter & alpha_fe_filter] # & bs_filter & prog_filter 
    
    # ------------------ Filter Eccentricity and Energy and Apocenter ------------------
    ecc_filter = star_data['ecc_50'] > 0.85
    energy_filter = star_data['E_50'] < 0
    # Missing distance uncertainty and apocenter filter
    # apocenter_filter = star_data['R_ap'] > 5
    # dist_err_filter = apogee_data['DIST_ERR'] < 1.5

    # ------------------ Apply stage 2 filters ------------------
    star_data = star_data[ecc_filter & energy_filter] # & apocenter_filter & dist_err_filter

    # ------------------ SQL-Based Gaia Distance Query ------------------
    if SQL:
        logging.info("Querying Gaia for distances...")

        # Extract Gaia IDs
        gaia_ids = np.array(star_data[gaia_id_col])
        # Set size for SQL query and split up GAIA IDs
        query_size = 750
        indiv_queries = np.array_split(gaia_ids, np.ceil(len(gaia_ids) / query_size))

        # Empty list to store the results of each query
        # Track missing GAIA IDs
        list_query_results = []
        missing_ids_set = set()

        # Query Gaia in chunks
        for i, query in enumerate(tqdm(indiv_queries, desc="Processing Queries")):
            gaia_id_list = ", ".join(query.astype(str))
            
            # Define SQL query
            distance_query = f"""
            SELECT source_id, r_med_photogeo, r_lo_photogeo, r_hi_photogeo
            FROM external.gaiaedr3_distance
            WHERE source_id IN ({gaia_id_list});
            """

            # Run query
            job = Gaia.launch_job(distance_query)
            results = job.get_results()

            # Store missing IDs
            query_ids = set(query)
            returned_ids = set(results['source_id'])
            missing_ids_set.update(query_ids - returned_ids)

            # Append results
            list_query_results.append(results)

        # Combine results
        all_query_results = vstack(list_query_results)
        missing_gaia_ids = np.array(list(missing_ids_set))

        # Remove stars with missing GAIA Data
        missing_ids_position = np.isin(gaia_ids, missing_gaia_ids)
        star_data = Table(star_data[~missing_ids_position])


        # Sort tables by GAIA ID's
        all_query_results.sort('source_id')
        star_data.sort(gaia_id_col)

        # Check if the GAIA ID's match before merging
        if not np.array_equal(star_data[gaia_id_col], all_query_results['source_id']):
            raise ValueError("Mismatch in GAIA IDs - Ensure the IDs match before merging.")

        # Ensure all_query_results is an Astropy Table
        all_query_results = Table(all_query_results)
        
        # Merge Gaia distances into the main dataset
        star_data['r_med_photogeo'] = all_query_results['r_med_photogeo']
        star_data['r_lo_photogeo'] = all_query_results['r_lo_photogeo']
        star_data['r_hi_photogeo'] = all_query_results['r_hi_photogeo']

        # ------------------ Save distance errors if data from sql provided ------------------
        dist_err_filter_hi = (star_data['r_hi_photogeo'] - star_data['r_med_photogeo']) < 1500
        dist_err_filter_lo = (star_data['r_med_photogeo'] - star_data['r_lo_photogeo']) < 1500
        star_data = star_data[dist_err_filter_hi & dist_err_filter_lo]

    # ------------------ Save filtered data if path provided ------------------
    # Store final number of stars
    final_star_count = len(star_data)
    logging.info(f"Final number of stars: {final_star_count}")
    logging.info(f"Fraction retained: {final_star_count / initial_star_count:.2%}")

    if save_path:
        star_data.write(save_path, format="fits", overwrite=True)
        logging.info(f"Filtered dataset saved to {save_path}")

    print("\n=== Filter Diagnostics: Stars Rejected by Each Criterion ===")

    # Convert to full table if not already
    original_data = convert_to_astropy_table(star_data_in)
    N_initial = len(original_data)

    # Diagnostic counts for each mask
    filters = {
        "Main Red Stars (extratarg == 0)": mrs_filter,
        "logg < 3.0": rg_filter,
        "[Fe/H] quality": fe_h_filter,
        "[Al/Fe] quality": al_fe_filter,
        "[Ce/Fe] quality": ce_fe_filter,
        "[Mg/Mn] quality": mg_mn_filter,
        "[alpha/Fe] quality": alpha_fe_filter,
        "Eccentricity > 0.85": ecc_filter,
        "Energy < 0": energy_filter,
    }

    for name, mask in filters.items():
        n_failed = N_initial - np.sum(mask)
        print(f"{name:30s} → {n_failed:4d} stars removed")
    
    # SQL-based filtering
    if SQL:
        n_sql_failed = len(star_data) + np.sum(missing_ids_position) - np.sum(dist_err_filter_hi & dist_err_filter_lo)
        print(f"Gaia SQL distance cut           → {n_sql_failed:4d} stars removed")

    return star_data   