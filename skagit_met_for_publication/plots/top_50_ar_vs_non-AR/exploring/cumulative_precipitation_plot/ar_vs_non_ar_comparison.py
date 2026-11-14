"""
Compare cumulative precipitation and streamflow between AR and non-AR events.

Creates side-by-side plots showing median cumulative precipitation and cumulative
discharge with 25%-75% quantile ranges shaded for both AR and non-AR event groups.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import xarray as xr
import geopandas as gpd
import regionmask
import zipfile
import tempfile
import rioxarray

# # Add parent directory to path to import plot_specific_ar_events
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from plot_specific_ar_and_non_ar_events.plot_specific_ar_events import extract_event_window, load_usgs_rdb

VAULT_DIR = "/data0/skagit_met/data_transfer/data"
# Hydrology data
BASE_DIR = "/data0/hernanqd/plots_code/skagit-met/skagit_met_for_publication"
HUC8_GEO = os.path.join(BASE_DIR, "data/GIS/SkagitSubBasin_HUC8.geojson")
HYDRO_EXP_DATA_DIR = os.path.join(BASE_DIR, "data")
HYDRO_Q_PATH = os.path.join(HYDRO_EXP_DATA_DIR, "usgs_12200500_discharge.rdb")

# Drainage area (square miles)
DRAINAGE_AREA_MI2 = 3093 #mi^2 extracted from the USGS website
DRAINAGE_AREA_FT2 = DRAINAGE_AREA_MI2 * 27_878_400

# Configuration
OUTPUT_DIR = os.path.join(BASE_DIR, "plots/top_50_ar_vs_non-AR/cumulative_precipitation_plot")

# Event dates from the comparison notebook
# NON_AR_DATES = [
#     '2021-11-16', '2010-12-14', '1995-02-21', '2015-11-14', '2020-01-08',
#     '1988-04-08', '1990-10-06', '2011-01-18', '2020-01-09', '2011-11-25',
#     '2014-01-14', '2004-12-12', '2017-12-20', '1995-11-26', '2020-02-03',
#     '2014-03-06', '1994-10-29', '1996-11-29', '1991-12-07', '2014-02-19',
#     '2014-03-17', '1995-12-02', '1999-12-13', '1997-10-06', '2007-03-13',
#     '2014-03-04', '1996-11-28', '2016-10-10', '2021-02-23', '2020-02-08',
#     '2024-03-02', '2001-12-19', '2005-01-21', '1997-02-01', '2018-01-30',
#     '2012-10-21', '2014-10-23', '1981-02-20', '2011-11-28', '1990-11-14',
#     '1997-01-31', '2020-09-27', '2006-01-08', '2016-10-11', '2008-11-15',
#     '1982-12-18', '1993-03-24', '2022-11-01', '2007-03-14', '2024-01-11'
# ]

# NON_AR_DATES = ['2008-07-02', '2008-07-03', '2008-07-04', '2012-07-10', '2012-07-15', '2020-06-01', '2022-07-04', '2022-07-02', '2012-07-18', '1997-05-13', '1986-06-01', '2018-05-16', '1999-08-05', '1999-07-11', '1986-06-02', '1981-01-02', '2008-07-07', '1999-07-22', '1997-06-24', '2011-07-17', '2014-05-16', '1987-05-01', '2011-07-19', '2011-07-09', '1997-06-06', '1999-07-21', '1985-05-24', '2011-07-20', '2011-07-23', '2018-05-24', '2023-05-05', '2018-05-09', '2011-07-18', '2023-05-15', '2017-05-23', '1982-07-04', '1999-07-08', '2006-05-17', '2014-05-04', '1982-05-26', '1986-06-03', '2011-06-14', '2002-07-09', '1982-06-12', '1991-07-25', '2017-12-02', '1985-06-09', '2008-06-01', '1999-07-30', '2022-06-30'] #5d window and 2000 non-AR sample
# NON_AR_DATES = ['2008-07-04', '2012-07-14', '2021-06-30', '1982-06-18', '2002-06-15', '2012-07-16', '2021-07-01', '2002-06-16', '2012-07-17', '1999-06-17', '2002-06-14', '2006-05-18', '2011-07-03', '2012-06-22', '1986-06-01', '2002-07-13', '2002-07-12', '2011-06-07', '2012-07-05', '1999-07-12', '1999-06-18', '2012-05-15', '2018-05-10', '2022-07-05', '2017-06-01', '1991-01-16', '2012-07-07', '1997-06-06', '1982-06-15', '1999-07-25', '2011-07-23', '2002-06-18', '1985-05-25', '2014-05-25', '1982-07-04', '2021-06-05', '1999-07-26', '1986-06-03', '2011-06-14', '1997-06-25', '2012-05-01', '1990-07-12', '2011-07-16', '1999-07-17', '1990-06-11', '2008-07-08', '2017-12-03', '1997-06-26', '2011-06-06', '1999-07-18'] #3d window and 200 non-AR sample
# NON_AR_DATES = ['2021-11-19', '2008-07-02', '2020-05-31', '1995-12-04', '1982-06-21', '1990-11-17', '2012-07-09', '2021-06-30', '1985-06-08', '1997-05-16', '2012-06-24', '2002-06-15', '2017-05-30', '2006-05-19', '2021-06-03', '2011-07-07', '2022-07-03', '1999-06-17', '1997-06-05', '2012-05-16', '2021-12-06', '1986-06-01', '1991-07-04', '1999-05-25', '2002-07-13', '2023-05-17', '2018-05-16', '1999-08-05', '2011-06-07', '2007-07-07', '2014-06-29', '1997-06-24', '1991-01-16', '2012-05-25', '1991-02-12', '1987-05-01', '2025-12-23', '1997-04-21', '1982-07-15', '1984-01-09', '2013-06-21', '2009-06-03', '1990-07-13', '2012-06-06', '2022-06-14', '2013-11-19', '2000-06-29', '1991-07-25', '2017-12-02', '1989-06-06'] #new filtering (keep contigous dates as one event and assign the date with peak discharge) - peak discharge
# NON_AR_DATES = ['1991-03-04', '2012-12-18', '1992-12-22', '2007-02-21', '1983-03-31', '1997-02-20', '2020-03-31', '2018-04-16', '1985-06-08', '1982-03-13'] #, top 10: new filtering (first, 5 day window arroun ar, then keep contigous dates as one event and assign the date with peak prism_3d_tot) - peak precip
NON_AR_DATES = ['1991-03-04', '2012-12-18', '1992-12-22', '2007-02-21', '1983-03-31', '1997-02-20', '2020-03-31', '2018-04-16', '1985-06-08', '1982-03-13', '1990-01-27', '1994-03-19', '1986-02-17', '2021-12-24', '1991-01-01', '1991-11-27', '2010-01-02', '2018-02-18', '1988-03-07', '2011-11-14', '1998-01-07', '2004-09-18', '1982-11-18', '2013-11-19', '1997-04-22', '1983-11-27', '2008-02-01', '2006-04-15', '1988-11-12', '1990-03-10', '1981-04-10', '2000-01-10', '2002-03-21', '2001-05-01', '2015-12-21', '2009-04-14', '2003-02-22', '2000-03-19', '1992-06-14', '2008-12-29', '1984-03-14', '1984-12-23', '2013-03-21', '1990-09-01', '2016-12-30', '2014-02-25', '1996-12-22', '2023-02-08', '1985-04-01', '2016-12-12'] #top 50: new filtering (first, 5 day window arroun ar, then keep contigous dates as one event and assign the date with peak prism_3d_tot) - peak precip
# NON_AR_DATES = ['1991-03-04', '2014-03-04', '1990-01-29', '2012-12-18', '1991-01-16', '1992-12-22', '2007-02-21', '1983-03-31', '1997-02-20', '2020-03-31', '2018-04-16', '2010-01-03', '1985-06-08', '1982-03-13', '1994-03-19', '1986-02-17', '2021-12-24', '1991-01-01', '2020-01-13', '1991-11-27', '2018-02-18', '2017-03-04', '1988-03-07', '2011-11-14', '1998-01-07', '1997-04-21', '2004-09-16', '2006-11-24', '1982-11-18', '2013-11-19', '1983-11-27', '2012-02-27', '1991-12-13', '2008-02-01', '2006-04-15', '1988-11-12', '1990-03-10', '1981-04-10', '2000-01-10', '2002-03-21', '2001-05-01', '2015-12-21', '2009-01-02', '2009-04-14', '1984-05-12', '2003-02-22', '2000-03-19', '1992-06-14', '2013-02-24', '1984-03-14'] #top 50: new filtering (first, 3 day window arroun ar, then keep contigous dates as one event and assign the date with peak prism_3d_tot) - peak precip
# NON_AR_DATES = ['1990-11-25', '1982-02-15', '2003-10-23', '1988-10-17', '1985-11-03', '1994-12-01', '2010-12-13', '1997-10-31', '1986-11-25', '2014-01-13', '1990-10-05', '1996-02-08', '1981-02-18', '1991-04-05', '2015-12-09', '1984-01-26', '2016-10-16', '2015-11-18', '2008-11-09', '2022-03-01', '1994-12-28', '1985-10-25', '1990-01-10', '2007-03-25', '1992-01-25', '1989-11-12', '2021-11-29', '1983-11-05', '2023-11-05', '1998-11-23', '2009-10-18', '2006-02-01', '1993-01-26', '2009-11-17', '1994-12-18', '2007-01-03', '2012-11-01', '2020-10-12', '1996-04-25', '1987-12-04', '2007-01-05', '1995-12-13', '2012-10-17', '2021-01-04', '1989-04-07', '2014-12-12', '1983-01-05', '2004-08-26', '1997-12-19', '2017-12-30'] #top 50: new filtering taking min ar_scale(do not remove non-ar arround ar, then keep contigous dates as one event and assign the date with peak prism_3d_tot) - peak precip

# NON_AR_DATES = ['2022-03-01', '1996-04-25', '2004-08-26', '2017-12-30', '2024-02-29', '1997-09-17', '2014-02-17', '1982-02-01', '1988-11-22', '2012-03-15', '2001-12-01', '2017-03-15', '2019-12-20', '1994-02-16', '2017-03-17', '2012-03-12', '1998-12-02', '2011-10-12', '1982-01-16', '2014-02-14', '2011-03-15', '2014-02-12', '2016-11-26', '2007-12-24', '1999-03-02', '1988-09-25', '2024-06-04', '2017-03-29', '1998-03-24', '2017-03-10', '2003-11-11', '1998-01-14', '2000-09-10', '1981-06-19', '1988-02-15', '2001-02-01', '1990-02-03', '2014-01-03', '2019-01-23', '2024-08-26', '2007-04-28', '1981-06-21', '1999-12-08', '1988-09-28', '2024-01-20', '2012-04-02', '2008-01-05', '1993-11-21', '2009-12-21', '2004-08-28'] #top 50: new filtering taking min ar_scale(first, 5 day window arround ar, then keep contigous dates as one event and assign the date with peak prism_3d_tot) - peak precip
# NON_AR_DATES = ['2019-12-23', '1997-09-18', '2014-02-19', '2024-03-02', '1990-01-29', '2012-03-16', '2011-03-16', '2007-02-21', '1982-02-01', '2012-12-07', '1997-02-20', '1998-01-17', '1985-06-08', '1991-01-01', '2020-01-13', '2024-06-05', '2007-12-25', '1998-12-02', '1999-01-20', '1991-11-26', '2017-03-19', '2023-12-11', '2011-10-12', '2011-11-14', '2011-04-06', '1982-01-16', '2004-09-17', '2013-11-19', '1997-04-22', '2021-12-25', '2000-01-10', '1984-11-30', '1988-03-09', '2006-04-16', '2001-11-01', '2003-02-22', '1999-03-02', '1981-04-11', '1985-11-08', '1996-12-22', '2019-01-24', '1994-02-18', '1990-08-31', '1995-01-19', '2018-04-18', '2001-12-04', '2009-12-22', '2001-02-05', '1993-11-22', '2024-12-30'] # final (5d window + filling)

# AR_DATES = [
#     '1990-11-24', '2006-11-07', '2009-01-08', '1990-11-25', '1990-11-11',
#     '1995-11-30', '1990-11-10', '2003-11-19', '1995-11-10', '2003-10-22',
#     '1989-11-11', '2003-10-21', '2003-10-18', '1982-02-15', '1997-03-20',
#     '1989-11-10', '1995-11-29', '1986-01-19', '1983-01-10', '1995-11-08',
#     '1982-02-16', '2006-11-06', '2021-11-15', '1986-11-24', '1988-10-16',
#     '2005-01-19', '2003-10-17', '1986-02-25', '2003-10-23', '2007-12-04',
#     '1988-10-17', '1990-11-12', '2014-11-28', '2003-10-19', '1985-11-03',
#     '1998-11-15', '1989-11-09', '1995-02-20', '1995-02-19', '2009-01-10',
#     '1994-12-01', '2005-01-20', '2002-02-22', '1982-01-25', '2006-11-05',
#     '1982-01-24', '1998-11-16', '2009-01-07', '2011-01-17', '2010-12-13'
# ]

# AR_DATES = ['1990-11-24', '2006-11-07', '2009-01-08', '1990-11-25', '1990-11-11', '1995-11-30', '1990-11-10', '2003-11-19', '1995-11-10', '2003-10-22', '1989-11-11', '2003-10-21', '2003-10-18', '1982-02-15', '1997-03-20', '1989-11-10', '1995-11-29', '1986-01-19', '1983-01-10', '1995-11-08', '1982-02-16', '2006-11-06', '2021-11-15', '1986-11-24', '1988-10-16', '2005-01-19', '2003-10-17', '1986-02-25', '2003-10-23', '2007-12-04', '1988-10-17', '1990-11-12', '2014-11-28', '2003-10-19', '1985-11-03', '1998-11-15', '1989-11-09', '1995-02-20', '1995-02-19', '2009-01-10', '1994-12-01', '2005-01-20', '2002-02-22', '1982-01-25', '2006-11-05', '1982-01-24', '1998-11-16', '2009-01-07', '2011-01-17', '2010-12-13']
# AR_DATES = ['1990-11-25', '1995-11-30', '1990-11-11', '2006-11-07', '2003-10-22', '2025-12-12', '2003-10-21', '2021-11-15', '1989-11-11', '2010-12-13', '1984-01-05', '2003-10-23', '2025-12-17', '2017-11-23', '1999-11-13', '2011-01-17', '1999-11-14', '2007-03-25', '1997-03-20', '2002-01-08', '2015-11-18', '2009-01-08', '2014-11-28', '2007-12-04', '2021-11-29', '2004-12-11', '2009-11-17', '1986-11-24', '2003-11-19', '2021-10-29', '2015-12-09', '2021-12-01', '2001-11-15', '1986-01-19', '2005-01-19', '1997-07-09', '1989-11-12', '1986-02-25', '2008-05-18', '2007-03-12', '2008-11-13', '2015-02-07', '2016-02-16', '2009-11-18', '1983-11-16', '1995-11-12', '2020-02-01', '1985-11-03', '1999-11-15', '2017-11-25'] #new filtering (keep contigous dates as one event and assign the date with peak discharge) - preak discharge
# AR_DATES = ['1990-11-24', '2006-11-07', '2009-01-08', '1990-11-11', '1995-11-30', '2003-11-19', '1995-11-10', '2003-10-22', '1989-11-11', '2003-10-21'] #top 10: new filtering (first, 5 day window arroun ar, then keep contigous dates as one event and assign the date with peak prism_3d_tot) - peak precip
AR_DATES = ['1990-11-24', '2006-11-07', '2009-01-08', '1990-11-11', '1995-11-30', '2003-11-19', '1995-11-10', '2003-10-22', '1989-11-11', '2003-10-21', '1982-02-15', '1997-03-20', '1986-01-19', '1983-01-10', '1995-11-08', '2021-11-15', '1986-11-24', '1988-10-16', '2005-01-19', '1986-02-25', '2003-10-23', '2007-12-04', '2014-11-28', '1985-11-03', '1998-11-15', '1995-02-20', '2009-01-10', '1994-12-01', '2002-02-22', '1982-01-25', '1982-01-24', '2011-01-17', '2010-12-13', '1997-10-31', '1998-12-30', '2004-12-11', '1986-11-25', '2017-11-23', '2014-01-13', '2002-01-08', '1988-04-06', '2021-10-29', '1997-01-01', '1990-10-05', '1996-02-08', '1986-11-20', '1999-11-13', '2022-12-27', '1985-10-27', '1981-02-18'] #top 50: new filtering (first, 5 day window arroun ar, then keep contigous dates as one event and assign the date with peak prism_3d_tot) - peak precip
# AR_DATES = ['1990-11-24', '2006-11-07', '2009-01-08', '1990-11-11', '1995-11-30', '2003-11-19', '1995-11-10', '2003-10-22', '1989-11-11', '2003-10-21', '1982-02-15', '1997-03-20', '1986-01-19', '1983-01-10', '1995-11-08', '2021-11-15', '1986-11-24', '1988-10-16', '2005-01-19', '1986-02-25', '2003-10-23', '2007-12-04', '2014-11-28', '1985-11-03', '1998-11-15', '1995-02-20', '2009-01-10', '1994-12-01', '2002-02-22', '1982-01-25', '1982-01-24', '2011-01-17', '2010-12-13', '1997-10-31', '1998-12-30', '2004-12-11', '1986-11-25', '2017-11-23', '2014-01-13', '2002-01-08', '1988-04-06', '2021-10-29', '1997-01-01', '1990-10-05', '1996-02-08', '1986-11-20', '1999-11-13', '2022-12-27', '1985-10-27', '1981-02-18'] #top 50: new filtering (first, 3 day window arroun ar, then keep contigous dates as one event and assign the date with peak prism_3d_tot) - peak precip

# AR_DATES = ['1990-11-24', '2006-11-07', '2009-01-08', '1990-11-11', '1995-11-30', '2003-11-19', '1995-11-10', '2003-10-22', '1989-11-11', '2003-10-21', '2003-10-18', '1997-03-20', '1986-01-19', '1983-01-10', '1995-11-08', '2021-11-15', '1986-11-24', '1988-10-16', '2005-01-19', '1986-02-25', '2007-12-04', '2014-11-28', '1998-11-15', '1995-02-20', '2009-01-10', '2002-02-22', '1982-01-25', '1982-01-24', '2011-01-17', '1982-02-14', '1998-12-30', '2004-12-11', '2017-11-23', '2002-01-08', '1988-04-06', '2021-10-29', '1997-01-01', '1986-11-20', '1999-11-13', '2022-12-27', '1985-10-27', '1987-03-04', '2001-11-15', '1981-02-17', '1984-01-05', '2015-11-16', '1994-12-20', '2009-11-18', '2003-03-14', '1997-01-20'] #top 50: new filtering taking min ar_scale(first, 5 day window arroun ar, then keep contigous dates as one event and assign the date with peak prism_3d_tot) - peak precip
# AR_DATES = ['1990-11-24', '2006-11-07', '2009-01-08', '1995-11-30', '2003-11-19', '2003-10-22', '2003-10-21', '2003-10-18', '1997-03-20', '1989-11-10', '1986-01-19', '1983-01-10', '1995-11-08', '2021-11-15', '1986-11-24', '1988-10-16', '1986-02-25', '2007-12-04', '2014-11-28', '1995-02-20', '2005-01-20', '2002-02-22', '1982-01-24', '1998-11-16', '2011-01-17', '1982-02-14', '1998-12-30', '2004-12-11', '2017-11-23', '2002-01-08', '1988-04-06', '2021-10-29', '1997-01-01', '1986-11-20', '2022-12-27', '1985-10-27', '2001-11-15', '1981-02-17', '1984-01-05', '1999-11-14', '1994-12-20', '2003-03-14', '1997-01-20', '1983-11-16', '1984-11-03', '2006-01-12', '1998-11-26', '1998-12-28', '1984-01-25', '2018-11-04'] # final (5d window + filling)


def load_prism_from_new_format(date, prism_root):
    """Load PRISM data from new format (TIFF) zip file for a specific date"""
    date_str = date.strftime('%Y%m%d')
    year = date.year
    decade_folder = get_decade_folder(year)
    zip_path = os.path.join(prism_root, decade_folder, f"prism_ppt_us_25m_{date_str}.zip")

    if not os.path.exists(zip_path):
        return None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(tmpdir)
                for root, dirs, files in os.walk(tmpdir):
                    for f in files:
                        if f.endswith('.tif'):
                            filepath = os.path.join(root, f)
                            da = rioxarray.open_rasterio(filepath)
                            return da.squeeze().copy(deep=True)
    except Exception as e:
        pass

    return None


def load_daymet_from_netcdf(year):
    """Load Daymet data from netCDF file for a specific year"""
    daymet_nc_path = os.path.join(VAULT_DIR, "daymet_new_hq", f"prcp_{year}_subset.nc")

    if not os.path.exists(daymet_nc_path):
        return None

    try:
        ds = xr.open_dataset(daymet_nc_path)
        return ds
    except Exception as e:
        print(f"  Error loading Daymet netCDF for {year}: {e}")
        return None


def load_usgs_rdb(filepath, param_name):
    """Load USGS RDB file"""
    try:
        df = pd.read_csv(filepath, sep='\t', comment='#')
        df = df.iloc[1:].reset_index(drop=True)
        val_cols = [c for c in df.columns if '_00' in c and not c.endswith('_cd')]
        if not val_cols: return pd.DataFrame(columns=['date', param_name])
        val_col = val_cols[0]
        df = df[['datetime', val_col]]
        df.columns = ['date', param_name]
        df['date'] = pd.to_datetime(df['date'])
        df[param_name] = pd.to_numeric(df[param_name], errors='coerce')
        return df
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return pd.DataFrame(columns=['date', param_name])


def load_regions():
    gdf = gpd.read_file(HUC8_GEO).to_crs("EPSG:4326")
    return gdf


def get_mask(gdf, lon, lat):
    mask = regionmask.mask_3D_geopandas(gdf, lon, lat)
    region_indices = []
    for r in ["Upper Skagit", "Sauk", "Lower Skagit"]:
        idx = gdf[gdf["Name"].str.contains(r, case=False)].index[0]
        region_indices.append(idx)
    return mask.sel(region=region_indices)


def calculate_basin_mean(da, mask_2d):
    data = da.values
    mask = mask_2d.values if hasattr(mask_2d, 'values') else mask_2d

    if data.ndim == 3:
        time_steps = data.shape[0]
        data_flat = data.reshape(time_steps, -1)
        mask_flat = mask.flatten()
        mask_idx = mask_flat > 0
        if mask_idx.any():
            mean_vals = np.nanmean(data_flat[:, mask_idx], axis=1)
        else:
            mean_vals = np.full(time_steps, np.nan)
        time_coord = da.time.values if 'time' in da.coords else range(time_steps)
        return pd.Series(mean_vals, index=pd.to_datetime(time_coord))
    elif data.ndim == 2:
        # Mask out nodata values (typically -9999 or negative values for TIFF)
        data = data.astype(float)
        data[data < 0] = np.nan  # Treat negative values as nodata

        data_flat = data.flatten()
        mask_flat = mask.flatten()
        mask_idx = mask_flat > 0
        if mask_idx.any():
            mean_val = np.nanmean(data_flat[mask_idx])
        else:
            mean_val = np.nan
        return mean_val
    else:
        raise ValueError(f"Unexpected data dimensions: {data.ndim}")


def get_decade_folder(year: int) -> str:
    if 2021 <= year <= 2024:
        return "2021-2024"
    start_decade = (year // 10) * 10
    if start_decade == 1980 and year > 1980:
        return "1981-1990"
    if year % 10 == 0 and year > 1980:
        return f"{start_decade-9}-{start_decade}"
    if start_decade % 10 == 0:
        return f"{start_decade+1}-{start_decade+10}"
    return f"{start_decade}-{(year//10)*10+9}"



def extract_event_window(event_date, products_to_extract=['prism', 'pnnl', 'daymet', 'conus', 'ucla', 'gridmet'], start_date=None, end_date=None):
    """Extract precipitation window for an event. Default is 8-day window (T-2 to T+5), but can use custom range."""

    if start_date and end_date:
        # Custom date range
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        window_dates = pd.date_range(start_dt, end_dt, freq='D').tolist()
    else:
        # Default 9-day window (T-3 to T+5)
        window_dates = []
        for i in range(-3, 6):
            window_dates.append(event_date + pd.Timedelta(days=i))

    year = event_date.year
    results = {f'{date.strftime("%Y-%m-%d")}': {} for date in window_dates}

    gdf = load_regions()
    masks_2d = {}

    try:
        sample_prism = xr.open_zarr(os.path.join(VAULT_DIR, "PRISM/1991-2000/1996-01-01_1996-12-31_daily_4km_PRISM_data.zarr"), consolidated=False)
        masks_2d['PRISM'] = get_mask(gdf, sample_prism.lon, sample_prism.lat).any(dim='region')
        sample_prism.close()
    except:
        pass

    try:
        sample_geo = xr.open_dataset(os.path.join(VAULT_DIR, "PNNL/SERDP6km.geo_em.d01.nc"))
        masks_2d['PNNL'] = get_mask(gdf, sample_geo.XLONG_M.values[0], sample_geo.XLAT_M.values[0]).any(dim='region')
    except:
        pass

    # Daymet mask will be created dynamically in extraction (from actual data coordinates)

    try:
        sample_conus = xr.open_zarr(os.path.join(BASE_DIR, "data/weather_data/conus404_skagit_precip_daily_full.zarr"))
        masks_2d['CONUS404'] = get_mask(gdf, sample_conus.lon, sample_conus.lat).any(dim='region')
        sample_conus.close()
    except:
        pass

    try:
        ucla_static = xr.open_dataset(os.path.join(VAULT_DIR, "ucla_era5_d02_daily/static/wrfinput_d02_coord.nc"))
        masks_2d['UCLA'] = get_mask(gdf, ucla_static.lon2d.values, ucla_static.lat2d.values).any(dim='region')
    except:
        pass

    try:
        sample_gridmet = xr.open_zarr(os.path.join(VAULT_DIR, "gridmet/2021_daily_4km_gridMET_data.zarr"))
        masks_2d['GridMET'] = get_mask(gdf, sample_gridmet.lon, sample_gridmet.lat).any(dim='region')
    except:
        pass

    # Extract PRISM from new format (TIFF)
    if 'prism' in products_to_extract:
        prism_root = os.path.join(VAULT_DIR, "prism_new_hq")
        prism_data = {}

        for date in window_dates:
            try:
                da = load_prism_from_new_format(date, prism_root)
                if da is None:
                    continue
                lon = da.x.values
                lat = da.y.values
                m_prism = get_mask(gdf, lon, lat).any(dim='region')
                mean_val = calculate_basin_mean(da, m_prism)
                prism_data[date.normalize()] = mean_val
            except Exception as date_error:
                pass

        # Add extracted values to results
        for date in window_dates:
            normalized_date = date.normalize()
            if normalized_date in prism_data:
                results[f'{date.strftime("%Y-%m-%d")}']['prism'] = float(prism_data[normalized_date])

        # Debug: Print what was found
        prism_count = sum(1 for d in prism_data.values() if not np.isnan(d))
        if prism_count > 0:
            print(f"  ✓ PRISM: {prism_count}/{len(window_dates)} dates extracted")

    # OLD CODE: Extract PRISM from zip files (commented out - now using new format first)
    # if 'prism' in products_to_extract:
    #     prism_data = {}
    #     # Load PRISM data for each date in the window from individual zip files
    #     for date in window_dates:
    #         try:
    #             ds = load_prism_from_zip(date, VAULT_DIR)
    #
    #             if ds is not None:
    #                 # Get precipitation variable (handle both 'ppt' and 'Band1')
    #                 ppt_var = 'ppt' if 'ppt' in ds.data_vars else 'Band1'
    #                 if ppt_var in ds.data_vars:
    #                     da = ds[ppt_var]
    #
    #                     # Handle dimensions: might be (lat, lon) or have time/band dims
    #                     if da.ndim == 3:  # (band, lat, lon) or (time, lat, lon)
    #                         da = da.isel(band=0) if 'band' in da.dims else da.isel(time=0)
    #                     elif da.ndim == 0:  # scalar, skip
    #                         ds.close()
    #                         continue
    #
    #                     # Create mask based on this file's actual grid dimensions
    #                     m_prism = get_mask(gdf, ds.lon.values, ds.lat.values).any(dim='region')
    #
    #                     # Calculate basin mean
    #                     if m_prism is not None:
    #                         mean_val = calculate_basin_mean(da, m_prism)
    #                         prism_data[date.normalize()] = mean_val
    #
    #                 ds.close()
    #         except Exception as date_error:
    #             pass  # Skip dates where zip file doesn't exist
    #
    #     # Add extracted values to results
    #     for date in window_dates:
    #         normalized_date = date.normalize()
    #         if normalized_date in prism_data:
    #             results[f'{date.strftime("%Y-%m-%d")}']['prism'] = float(prism_data[normalized_date])

    # OLD CODE: Extract PRISM from zarr (commented out - now loading from zip files)
    # if 'prism' in products_to_extract:
    #     try:
    #         dec = get_decade_folder(year)
    #         prism_path_4k = os.path.join(VAULT_DIR, "PRISM", dec, f"{year}-01-01_{year}-12-31_daily_4km_PRISM_data.zarr")
    #         prism_path_800 = os.path.join(VAULT_DIR, "prism_ppt_800m", f"{year}-01-01_{year}-12-31_daily_800m_PRISM_data.zarr")
    #         prism_path = prism_path_4k if os.path.exists(prism_path_4k) else prism_path_800
    #         if os.path.exists(prism_path):
    #             ds = xr.open_zarr(prism_path, consolidated=False)
    #             ds['time'] = pd.to_datetime(ds.time.values).normalize()
    #             da = ds['ppt'].sel(time=window_dates, method='nearest')
    #             if "800m" in prism_path:
    #                 m_prism = get_mask(gdf, ds.lon, ds.lat).any(dim='region')
    #             else:
    #                 m_prism = masks_2d.get('PRISM')
    #             if m_prism is not None:
    #                 prism_daily = calculate_basin_mean(da, m_prism)
    #                 prism_daily.index = prism_daily.index.normalize()
    #                 prism_daily = prism_daily[~prism_daily.index.duplicated(keep='first')]
    #                 for date in window_dates:
    #                     normalized_date = date.normalize()
    #                     if normalized_date in prism_daily.index:
    #                         val = prism_daily[normalized_date]
    #                         results[f'{date.strftime("%Y-%m-%d")}']['prism'] = float(val) if not isinstance(val, pd.Series) else float(val.iloc[0])
    #             ds.close()
    #     except Exception as e:
    #         print(f"Error loading PRISM for {event_date.strftime('%Y-%m-%d')}: {e}")

    # Extract PNNL
    if 'pnnl' in products_to_extract and year <= 2020:
        try:
            pnnl_path = os.path.join(VAULT_DIR, "PNNL/historical", str(year), f"PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.{year}.nc")
            if os.path.exists(pnnl_path):
                ds = xr.open_dataset(pnnl_path)
                hourly_dates = []
                for d in window_dates:
                    hourly_dates.extend(pd.date_range(d, d + pd.Timedelta(hours=23), freq='h'))
                available_hourly = [d for d in hourly_dates if d in ds.time.values]
                if available_hourly and 'PNNL' in masks_2d:
                    da_subset = ds['PREC_ACC_NC'].sel(time=available_hourly)
                    s_hourly = calculate_basin_mean(da_subset, masks_2d['PNNL'])
                    pnnl_daily = s_hourly.resample('1D').sum()
                    pnnl_daily.index = pnnl_daily.index.normalize()
                    for date in window_dates:
                        normalized_date = date.normalize()
                        if normalized_date in pnnl_daily.index:
                            val = pnnl_daily[normalized_date]
                            results[f'{date.strftime("%Y-%m-%d")}']['pnnl'] = float(val) if not isinstance(val, pd.Series) else float(val.iloc[0])
                ds.close()
        except Exception as e:
            print(f"Error loading PNNL for {event_date.strftime('%Y-%m-%d')}: {e}")

    # Extract Daymet (new format - netCDF files)
    if 'daymet' in products_to_extract:
        try:
            ds = load_daymet_from_netcdf(year)
            if ds is not None:
                ds['time'] = pd.to_datetime(ds.time.values).normalize()
                da = ds['prcp']

                # Create Daymet mask dynamically if not cached
                if 'Daymet' not in masks_2d:
                    lon_2d = ds['lon'].values
                    lat_2d = ds['lat'].values
                    masks_2d['Daymet'] = get_mask(gdf, lon_2d, lat_2d).any(dim='region')

                da_window = da.sel(time=window_dates, method='nearest')
                daymet_daily = calculate_basin_mean(da_window, masks_2d['Daymet'])
                daymet_daily.index = daymet_daily.index.normalize()
                daymet_daily = daymet_daily[~daymet_daily.index.duplicated(keep='first')]
                for date in window_dates:
                    normalized_date = date.normalize()
                    if normalized_date in daymet_daily.index:
                        val = daymet_daily[normalized_date]
                        results[f'{date.strftime("%Y-%m-%d")}']['daymet'] = float(val) if not isinstance(val, pd.Series) else float(val.iloc[0])
                ds.close()
        except Exception as e:
            print(f"Error loading Daymet for {event_date.strftime('%Y-%m-%d')}: {e}")

    # Extract CONUS404
    if 'conus' in products_to_extract:
        try:
            conus_path = os.path.join(BASE_DIR, "data/weather_data/conus404_skagit_precip_daily_full.zarr")
            if os.path.exists(conus_path):
                ds = xr.open_zarr(conus_path)
                ds['time'] = pd.to_datetime(ds.time.values).normalize()
                da = ds['precip_daily'].sel(time=window_dates, method='nearest')
                if 'CONUS404' in masks_2d:
                    conus_daily = calculate_basin_mean(da, masks_2d['CONUS404'])
                    conus_daily.index = conus_daily.index.normalize()
                    conus_daily = conus_daily[~conus_daily.index.duplicated(keep='first')]
                    for date in window_dates:
                        normalized_date = date.normalize()
                        if normalized_date in conus_daily.index:
                            val = conus_daily[normalized_date]
                            results[f'{date.strftime("%Y-%m-%d")}']['conus'] = float(val) if not isinstance(val, pd.Series) else float(val.iloc[0])
                    if event_date.strftime('%Y-%m-%d') in ['1995-11-29', '1990-11-10', '2021-11-15']:
                        print(f"  [DEBUG-DAILY] {event_date.strftime('%Y-%m-%d')} CONUS:")
                        print(f"    Daily values: {dict(conus_daily)}")
                        print(f"    Cumulative: {dict(conus_daily.cumsum())}")
                ds.close()
        except Exception as e:
            print(f"Error loading CONUS404 for {event_date.strftime('%Y-%m-%d')}: {e}")

    # Extract UCLA
    if 'ucla' in products_to_extract:
        try:
            u_paths = [
                os.path.join(VAULT_DIR, "ucla_era5_d02_daily", "prec", f"prec.daily.era5.d02.{year-1}.nc"),
                os.path.join(VAULT_DIR, "ucla_era5_d02_daily", "prec", f"prec.daily.era5.d02.{year}.nc")
            ]
            series_list = []
            for p in u_paths:
                if os.path.exists(p):
                    try:
                        ds = xr.open_dataset(p)
                        u_var = "prec" if "prec" in ds.data_vars else "pr"
                        if 'day' in ds.dims:
                            ds = ds.rename({'day': 'time'})
                        ds['time'] = pd.to_datetime(ds.time.values).normalize()
                        available_dates = [d for d in window_dates if d in ds.time.values]
                        if available_dates and 'UCLA' in masks_2d:
                            da_subset = ds[u_var].sel(time=available_dates)
                            s = calculate_basin_mean(da_subset, masks_2d['UCLA'])
                            s.index = s.index.normalize()
                            series_list.append(s)
                        ds.close()
                    except:
                        pass
            if series_list:
                ucla_daily = pd.concat(series_list).sort_index()
                ucla_daily = ucla_daily[~ucla_daily.index.duplicated(keep='first')]
                for date in window_dates:
                    normalized_date = date.normalize()
                    if normalized_date in ucla_daily.index:
                        val = ucla_daily[normalized_date]
                        results[f'{date.strftime("%Y-%m-%d")}']['ucla'] = float(val) if not isinstance(val, pd.Series) else float(val.iloc[0])
        except Exception as e:
            print(f"Error loading UCLA for {event_date.strftime('%Y-%m-%d')}: {e}")

    # Extract GridMET
    if 'gridmet' in products_to_extract:
        try:
            gridmet_path = os.path.join(VAULT_DIR, f"gridmet/{year}_daily_4km_gridMET_data.zarr")
            if os.path.exists(gridmet_path):
                ds = xr.open_zarr(gridmet_path)
                if 'day' in ds.dims:
                    ds = ds.rename({'day': 'time'})
                ds['time'] = pd.to_datetime(ds.time.values).normalize()
                var = 'prcp' if 'prcp' in ds.data_vars else 'precipitation_amount'
                da = ds[var].sel(time=window_dates, method='nearest')
                if 'GridMET' in masks_2d:
                    gridmet_daily = calculate_basin_mean(da, masks_2d['GridMET'])
                    gridmet_daily.index = gridmet_daily.index.normalize()
                    gridmet_daily = gridmet_daily[~gridmet_daily.index.duplicated(keep='first')]
                    for date in window_dates:
                        normalized_date = date.normalize()
                        if normalized_date in gridmet_daily.index:
                            val = gridmet_daily[normalized_date]
                            results[f'{date.strftime("%Y-%m-%d")}']['gridmet'] = float(val) if not isinstance(val, pd.Series) else float(val.iloc[0])
                ds.close()
        except Exception as e:
            print(f"Error loading GridMET for {event_date.strftime('%Y-%m-%d')}: {e}")

    return pd.DataFrame(results).T


def load_usgs_rdb(filepath, param_name):
    """Load USGS RDB file"""
    try:
        df = pd.read_csv(filepath, sep='\t', comment='#')
        df = df.iloc[1:].reset_index(drop=True)
        val_cols = [c for c in df.columns if '_00' in c and not c.endswith('_cd')]
        if not val_cols: return pd.DataFrame(columns=['date', param_name])
        val_col = val_cols[0]
        df = df[['datetime', val_col]]
        df.columns = ['date', param_name]
        df['date'] = pd.to_datetime(df['date'])
        df[param_name] = pd.to_numeric(df[param_name], errors='coerce')
        return df
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return pd.DataFrame(columns=['date', param_name])


def extract_all_events(event_dates, products=['prism', 'pnnl', 'daymet', 'conus', 'ucla', 'gridmet']):
    """Extract cumulative precipitation for all events in a list."""
    all_data = {}

    for event_date_str in event_dates:
        event_date = pd.to_datetime(event_date_str)
        try:
            window_df = extract_event_window(event_date, products_to_extract=products)
            if window_df is not None and len(window_df) > 0:
                all_data[event_date_str] = window_df
        except Exception as e:
            print(f"  Error extracting {event_date_str}: {e}")

    return all_data


def compute_cumulative_stats(all_event_data, product='prism'):
    """Compute median and min-max cumulative precipitation across events.

    Returns statistics indexed by relative days from event date (day 0 = event date).
    """
    cumulative_series = []

    for event_date_str, window_df in all_event_data.items():
        if product in window_df.columns:
            event_date = pd.to_datetime(event_date_str)
            window_df.index = pd.to_datetime(window_df.index)
            window_df = window_df.sort_index()

            # Convert dates to relative days from event date
            relative_days = (window_df.index - event_date).days

            cumsum = window_df[product].cumsum()
            cumsum.index = relative_days
            cumulative_series.append(cumsum)

    if not cumulative_series:
        return None, None, None

    # Align all series to common relative day index
    combined = pd.concat(cumulative_series, axis=1)

    # Calculate statistics
    median = combined.median(axis=1)
    min_val = combined.min(axis=1)
    max_val = combined.max(axis=1)

    # Sort by relative days
    median = median.sort_index()
    min_val = min_val.sort_index()
    max_val = max_val.sort_index()

    return median, min_val, max_val


def calculate_cumulative_discharge_mm(discharge_df):
    """Convert discharge in cfs to cumulative mm using trapezoidal integration."""
    if discharge_df.empty or 'discharge_cfs' not in discharge_df.columns:
        return pd.Series(dtype=float)

    discharge_fts = discharge_df['discharge_cfs'] / DRAINAGE_AREA_FT2
    discharge_ftd = discharge_fts * 60 * 60 * 24
    discharge_mmd = discharge_ftd * 304.8

    dates = discharge_df['date'].values
    date_numeric = (pd.to_datetime(dates) - pd.to_datetime(dates[0])).total_seconds() / (60 * 60 * 24)

    cum_discharge = cumulative_trapezoid(
        discharge_mmd.values,
        x=date_numeric.values,
        initial=0
    )

    return pd.Series(cum_discharge, index=discharge_df.index)


def compute_cumulative_discharge_stats(event_dates, q_df):
    """Compute median and min-max cumulative discharge across events.

    Returns statistics indexed by relative days from event date.
    """
    cumulative_series = []

    for event_date_str in event_dates:
        event_date = pd.to_datetime(event_date_str)

        # Extract 12-day window (T-3 to T+8)
        window_start = event_date - pd.Timedelta(days=3)
        window_end = event_date + pd.Timedelta(days=8)

        # Get discharge data for this window
        discharge_window = q_df[(q_df['date'] >= window_start) & (q_df['date'] <= window_end)]

        if not discharge_window.empty:
            # Calculate cumulative discharge
            cum_discharge = calculate_cumulative_discharge_mm(discharge_window)

            # Convert dates to relative days
            relative_days = (discharge_window['date'] - event_date).dt.days + (discharge_window['date'] - event_date).dt.seconds / (24 * 3600)
            cum_discharge.index = relative_days
            cumulative_series.append(cum_discharge)

    if not cumulative_series:
        return None, None, None

    # Align all series to common relative day index
    combined = pd.concat(cumulative_series, axis=1)

    # Calculate statistics
    median = combined.median(axis=1)
    min_val = combined.min(axis=1)
    max_val = combined.max(axis=1)

    # Sort by relative days
    median = median.sort_index()
    min_val = min_val.sort_index()
    max_val = max_val.sort_index()

    return median, min_val, max_val


def plot_ar_comparison(product='prism'):
    """Create comparison plot of AR vs non-AR events (precipitation + streamflow)."""
    print(f"Extracting data for non-AR events...")
    non_ar_data = extract_all_events(NON_AR_DATES)
    print(f"  Extracted {len(non_ar_data)} non-AR events")

    print(f"Extracting data for AR events...")
    ar_data = extract_all_events(AR_DATES)
    print(f"  Extracted {len(ar_data)} AR events")

    # Load discharge data
    print(f"Loading discharge data...")
    q_df = load_usgs_rdb(HYDRO_Q_PATH, 'discharge_cfs')

    # Compute precipitation statistics
    print(f"Computing cumulative precipitation statistics for {product.upper()}...")
    non_ar_precip_median, non_ar_precip_min, non_ar_precip_max = compute_cumulative_stats(non_ar_data, product)
    ar_precip_median, ar_precip_min, ar_precip_max = compute_cumulative_stats(ar_data, product)

    # Compute discharge statistics
    print(f"Computing cumulative discharge statistics...")
    non_ar_discharge_median, non_ar_discharge_min, non_ar_discharge_max = compute_cumulative_discharge_stats(NON_AR_DATES, q_df)
    ar_discharge_median, ar_discharge_min, ar_discharge_max = compute_cumulative_discharge_stats(AR_DATES, q_df)

    # Create side-by-side plots with shared y-axis
    fig, (ax_precip, ax_discharge) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # Color palette: categorical colors (AR in blue, non-AR in coral/orange)
    color_ar = '#1f77b4'      # Blue for AR
    color_non_ar = '#ff7f0e'  # Orange for non-AR

    # ===== PRECIPITATION PLOT =====
    # Plot non-AR events
    if non_ar_precip_median is not None:
        ax_precip.fill_between(
            non_ar_precip_median.index,
            non_ar_precip_min,
            non_ar_precip_max,
            alpha=0.25,
            color=color_non_ar,
            label='Non-AR (min-max)'
        )
        ax_precip.plot(
            non_ar_precip_median.index,
            non_ar_precip_median,
            linewidth=2.5,
            color=color_non_ar,
            marker='o',
            markersize=6,
            label='Non-AR (median)',
            alpha=0.9
        )

    # Plot AR events
    if ar_precip_median is not None:
        ax_precip.fill_between(
            ar_precip_median.index,
            ar_precip_min,
            ar_precip_max,
            alpha=0.25,
            color=color_ar,
            label='AR (min-max)'
        )
        ax_precip.plot(
            ar_precip_median.index,
            ar_precip_median,
            linewidth=2.5,
            color=color_ar,
            marker='s',
            markersize=6,
            label='AR (median)',
            alpha=0.9
        )

    ax_precip.set_xlabel('Days Relative to Event Date', fontsize=11, fontweight='bold')
    ax_precip.set_ylabel('Cumulative Precipitation (mm)', fontsize=11, fontweight='bold')
    ax_precip.set_title(f'Cumulative {product.upper()} Precipitation', fontsize=12, fontweight='bold', pad=15)
    ax_precip.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax_precip.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # ===== DISCHARGE PLOT =====
    # Plot non-AR events
    if non_ar_discharge_median is not None:
        ax_discharge.fill_between(
            non_ar_discharge_median.index,
            non_ar_discharge_min,
            non_ar_discharge_max,
            alpha=0.25,
            color=color_non_ar,
            label='Non-AR (min-max)'
        )
        ax_discharge.plot(
            non_ar_discharge_median.index,
            non_ar_discharge_median,
            linewidth=2.5,
            color=color_non_ar,
            marker='o',
            markersize=6,
            label='Non-AR (median)',
            alpha=0.9
        )

    # Plot AR events
    if ar_discharge_median is not None:
        ax_discharge.fill_between(
            ar_discharge_median.index,
            ar_discharge_min,
            ar_discharge_max,
            alpha=0.25,
            color=color_ar,
            label='AR (min-max)'
        )
        ax_discharge.plot(
            ar_discharge_median.index,
            ar_discharge_median,
            linewidth=2.5,
            color=color_ar,
            marker='s',
            markersize=6,
            label='AR (median)',
            alpha=0.9
        )

    ax_discharge.set_xlabel('Days Relative to Event Date', fontsize=11, fontweight='bold')
    ax_discharge.set_ylabel('Cumulative Discharge (mm)', fontsize=11, fontweight='bold')
    ax_discharge.set_title('Cumulative Streamflow (USGS 12200500)', fontsize=12, fontweight='bold', pad=15)
    ax_discharge.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax_discharge.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    fig.suptitle(
        f'Top 50 Events: AR vs Non-AR Comparison\n'
        f'Precipitation ({product.upper()}) and Streamflow (Median with Min-Max Range)',
        fontsize=14,
        fontweight='bold',
        y=1.00
    )

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f'ar_vs_non_ar_cumulative_{product}_with_discharge.png')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_ar_comparison(product='prism')
