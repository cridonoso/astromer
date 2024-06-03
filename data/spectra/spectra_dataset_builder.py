#!/usr/bin/python3

import urllib.request
import os.path
import pandas as pd
import numpy as np
import astropy
from astropy.io import fits

# The input file below was obtained by leveraging the SQL interface made available by the SDSS project
# More information can be obtained at: https://skyserver.sdss.org/dr18
# The list of SQL commands used to generate the initial version of the dataset used were:
#
# SELECT TOP 20000 specObjID,class,plateID,mjd,plate,fiberID,run2d into mydb.MyTable from SpecObj WHERE class='GALAXY' AND zWarning=0 AND targetType='SCIENCE';
# INSERT INTO mydb.MyTable select TOP 20000 specObjID,class,plateID,mjd,plate,fiberID,run2d from SpecObj WHERE class='QSO' AND zWarning=0 AND targetType='SCIENCE';
# INSERT INTO mydb.MyTable select TOP 20000 specObjID,class,plateID,mjd,plate,fiberID,run2d from SpecObj WHERE class='STAR' AND zWarning=0 AND targetType='SCIENCE';
#
# The table created was later downloaded as a csv file (inputfile below) and used to obtain the FITS files for each entry

fitsdir='fits/'
csvdir='csv/'
inputfile="mytable.csv"
spectradb = pd.read_csv(inputfile)

if not os.path.exists(fitsdir):
    os.makedirs(fitsdir)

if not os.path.exists(csvdir):
    os.makedirs(csvdir)

for index, row in spectradb.iterrows():
    specfilename=f'spec-{str(row["plate"]).zfill(4)}-{row["mjd"]}-{str(row["fiberID"]).zfill(4)}.fits'
    if not os.path.isfile(fitsdir + specfilename):
        # Obtaining based on instructions from https://dr18.sdss.org/optical/spectrum/view/data/access
        baseurl=f'http://dr17.sdss.org/sas/dr17/sdss/spectro/redux/{row["run2d"]}/spectra/lite/{str(row["plate"]).zfill(4)}/'
        urllib.request.urlretrieve(baseurl + specfilename, fitsdir + specfilename)

    specfits=fits.open(fitsdir + specfilename)

    data=np.array(specfits[1].data.tolist())[:,[0,1]]
    data[:,1]=np.power(10,data[:,1])
    data = np.fliplr(data)

    # specObjID is enough for us to get any extra data from object, and class will be used for classification
    outputfilename=f'spec-sdss-{str(index).zfill(7)}-{row["specObjID"]}-{row["class"]}.csv.gz'
    np.savetxt(csvdir + outputfilename, data, header="wavelength,flux", delimiter=",", comments="")
    specfits.close()


