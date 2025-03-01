import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from astropy.io import fits
from astropy.table import Table
from astropy.table import vstack
from astroquery.gaia import Gaia

from scipy.stats import norm

import logging
logging.basicConfig(level=logging.INFO)