#!/usr/bin/env python
""" 
Python script to download selected files from gdex.ucar.edu.
After you save the file, don't forget to make it executable
i.e. - "chmod 755 <name_of_script>"
"""
import sys, os
from urllib.request import build_opener

# Output directory configuration
output_dir = "/data0/skagit_met/data_transfer/data/CONUS_hourly/hourly_ar_non_ar_events"
os.makedirs(output_dir, exist_ok=True)

opener = build_opener()

filelist = [
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_00:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_01:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_02:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_03:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_04:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_05:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_06:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_07:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_08:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_09:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_10:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_11:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_12:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_13:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_14:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_15:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_16:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_17:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_18:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_19:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_20:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_21:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_22:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966075/966075.PREC_ACC_NC.wrf2d_d01_1995-12-05_23:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_00:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_01:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_02:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_03:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_04:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_05:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_06:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_07:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_08:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_09:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_10:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_11:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_12:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_13:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_14:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_15:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_16:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_17:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_18:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_19:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_20:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_21:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_22:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-16_23:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_00:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_01:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_02:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_03:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_04:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_05:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_06:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_07:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_08:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_09:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_10:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_11:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_12:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_13:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_14:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_15:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_16:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_17:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_18:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_19:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_20:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_21:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_22:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-17_23:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_00:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_01:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_02:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_03:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_04:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_05:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_06:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_07:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_08:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_09:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_10:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_11:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_12:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_13:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_14:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_15:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_16:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_17:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_18:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_19:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_20:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_21:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_22:00:00.nc',
  'https://request.gdex.ucar.edu/dsrqst/QUERBESDUHART966076/966076.PREC_ACC_NC.wrf2d_d01_1990-11-18_23:00:00.nc'
  ]

for file in filelist:
    ofile = os.path.basename(file)
    sys.stdout.write("downloading " + ofile + " ... ")
    sys.stdout.flush()
    infile = opener.open(file)
    outfile = open(os.path.join(output_dir, ofile), "wb")
    outfile.write(infile.read())
    outfile.close()
    sys.stdout.write("done\n")
