'''
Author == Alexander A. Massoud
Date   == 7/20/2021
Script == Remove telluric noise, fit and subtract continuum, and ID emission lines in table from Kaplan et al. (2017), all for ARCoIRIS NIR spectra
Input  == FITS files, variables (see VARIABLES section for explanation)
Output == unsubbed spectra, subbed spectra, IDed lines, and recreated Kaplan table, all in nested lists (i.e. list[file][order][nf*])
* nf only relevant index for lines and tables

Key    == #! : Variable to set
          USER INPUT, AUTOMATED : Block comment if you want the script to run automatically or with visual inspection
'''

'''
PACKAGES
'''
# handle matrix math and plotting
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

'''
# Remove block comment for publication-quality plots
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
'''

# text params for proposal
plt.rc('font', family='serif', size=17)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# handle FITS files and units
from astropy.io import fits
import astropy.units as u
from astropy.visualization import quantity_support
quantity_support() # for getting units on the axes with Astropy

# handle spectrum analysis
from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.manipulation import extract_region, noise_region_uncertainty
from specutils.fitting import fit_generic_continuum, find_lines_threshold
from specutils.analysis import line_flux, gaussian_sigma_width, gaussian_fwhm, fwhm, fwzi

# interface with a PostgreSQL database
import psycopg2

# create and manage tables
from astropy.table import QTable, Table, vstack

# handle normal fits
import scipy.stats
from scipy.integrate import quad
from astropy.modeling import models
from specutils.fitting import fit_lines

# handle linear fits
from scipy.stats import linregress

#handle stats
import statistics
import math

'''
FUNCTIONS
'''
def readFITSdata(FITS_list):
    '''
    Function == Load, read, and close processed ARCoIRIS FITS files
    Input    == List of FITS filenames
    Output   == Data of each FITS file
    '''
    data = ([])

    for file in range(len(FITS_list)):
        hdul = fits.open(FITS_list[file])
        data.append(hdul[0].data)           # NOTE: This points correctly for ARCoIRIS data
        hdul.close()

    return data

def trimNAN(data_list):
    '''
    Function == Remove NANs from the spectrum
    Input    == Nested list of ordered spectrum for each FITS file
    Output   == Nested list of ordered trimmed spectrum (No NANs so Spectrum1D objects can be built)
    '''
    trim = emptyFITSList(data_list)                         # create empty lists for each FITS file

    for file in range(len(data_list)):                      # range through each FITS file
        for order in range(len(data_list[file])):           # range through each spectral order
            trim[file].append([])                           # add an empty list to hold lamb and flux data

            order_flux_data = data_list[file][order][1]     # point to each order's flux
            order_lamb_data = data_list[file][order][0]     # point to each order's spectral axis

            flux_nan = np.isnan(order_flux_data)            # logical matrix that flags NANs in flux
            lamb_nan = np.isnan(order_lamb_data)            # logical matrix that flags NANs in spectral axis

            flux_cnt = 0                                    # count the amount of NANs
            lamb_cnt = 0

            for i in range(len(flux_nan)):                  # range through the logical matrix
                if flux_nan[i] == True :                    # flag NANS
                    flux_cnt += 1

            for i in range(len(lamb_nan)):                  # do the same for the spectral axis
                if lamb_nan[i] == True :
                    lamb_cnt += 1

            if flux_cnt > lamb_cnt :                        # trim based on which axis has more NANs
                nan_arr = np.logical_not(flux_nan)          # invert logic in NAN matrix
                axes = (order_lamb_data[nan_arr], order_flux_data[nan_arr]) # 2ple with trimmed spectral axis and flux
                trim[file][order].append(axes)              # append the 2ple into the order's empty list

            else:                                           # trim if lamb_cnt >= flux_cnt
                nan_arr = np.logical_not(lamb_nan)
                axes = (order_lamb_data[nan_arr], order_flux_data[nan_arr])
                trim[file][order].append(axes)

    return trim

def emptyFITSList(list):
    '''
    Function == Create a nested list of empty lists to hold Spectrum1D objects for each FITS file
    Input    == Nested list of ordered spectrum for each FITS file
    Output   == Nested list of empty lists of the same length (One per FITS file)
    '''
    empty = ([])
    for file in range(len(list)):   # range through the amount of FITS files
        empty.append([])

    return empty

def spec1D(trim_list):
    '''
    Function == Create Spectrum1D objects from FITS data without nan values
    UNITS    == (um, erg cm-2 s-1 AA-2)
    Input    == List of data from processed ARCoIRIS FITS files WIHOUT NAN VALUES
    Output   == Spectrum1D object for each FITS file
    '''
    spec = emptyFITSList(trim_list)                         # create empty lists for each FITS file

    for file in range(len(trim_list)):                      # range through each FITS file
        for order in range(len(trim_list[file])):           # range through each spectral order
            spec[file].append([])                           # add an empty list for each Spectrum1D object

            tple = trim_list[file][order][0]                # point to the data 2ple

            flux = tple[1]*u.Unit('erg cm-2 s-1 AA-2')      # attach units to flux
            lamb = tple[0]*u.um                             # attach units to spectral axis

            spec[file][order].append(Spectrum1D(flux=flux, spectral_axis=lamb)) # create and append Spectrum1D object

    return spec

def teluBgone(spec_list, region_4, region_3):
    '''
    Function == Extract the region of each relevant order without telluric corruption
    Input    == Nested list of Spectrum1D objects, spectral regions without telluric corruption
    Output   == Spectrum1D object for each relevant order without telluric corrutption
    '''
    telu = emptyFITSList(spec_list)                         # create empty lists for each FITS file

    for file in range(len(spec_list)):                      # range through each FITS file
        for order in range(len(spec_list[file])):           # range through each spectral order
            if order == 0 :                                 # Act on order 4...
                telu[file].append([])                       # add an empty list for each Spectrum1D object

                spectrum = spec_list[file][order][0]
                extracted_spectrum = extract_region(spectrum, region_4) # Extract Spec1D object using SpectralRegion

                telu[file][order].append(extracted_spectrum)# append Spectrum1D object

            elif order == 1 :                               # ... and order 3 ONLY (relevant orders)
                telu[file].append([])                       # add an empty list for each Spectrum1D object

                spectrum = spec_list[file][order][0]
                extracted_spectrum = extract_region(spectrum, region_3) # Extract Spec1D object using SpectralRegion

                telu[file][order].append(extracted_spectrum) # append Spectrum1D object

    return telu

def fitNsubList(spec1D_list):
    '''
    Function == Fit the continuum of each order and subtract it
    Input    == Nested list of Spectrum1D objects
    Output   == Nested list of continuum subtracted Spectrum1D objects
    '''
    fit = emptyFITSList(spec1D_list)                        # create empty lists for each FITS file

    for file in range(len(spec1D_list)):                    # range through each FITS file
        for order in range(len(spec1D_list[file])):         # range through each spectral order
            fit[file].append([])                            # add an empty list for each Spectrum1D object

            spec1D = spec1D_list[file][order][0]            # point to the Spectrum1D object
            spectral_axis = spec1D.spectral_axis            # point to the spectral axis

            g1_fit = fit_generic_continuum(spec1D)          # fit the order
            flux_continuum_fitted = g1_fit(spectral_axis)   # generate the continuum

            spec1D_subbed = spec1D - flux_continuum_fitted  # subtract the continuum

            fit[file][order].append(spec1D_subbed)          # append the subtracted order's spectra

    return fit

def lineFinding(spec1D_subbed, region_4, region_3, nf_list):
    '''
    Function == Find emission and absorption lines in each relevant order
    Input    == Nested list of continuum subbed Spec1D objects, spectral regions for each relevant order, noise factor
    Output   == Nested list of QTables that contain columns for each IDed feature based on different noise factors
    '''
    lines = emptyFITSList(spec1D_subbed)                    # create empty lists for each FITS file

    for file in range(len(spec1D_subbed)):                  # range through each FITS file
        for order in range(len(spec1D_subbed[file])):       # range through each spectral order
            if order == 0 :                                 # work on Order 4...
                lines[file].append([])                      # add an empty list for each QTable

                noise_region = region_4                     # Point to the region to determine noise
                spectrum = spec1D_subbed[file][order][0]    # Point to the Spec1D object
                spectrum = noise_region_uncertainty(spectrum, noise_region) # Estimate uncertainty

                for nf in range(len(nf_list)):              # range through each noise factor
                    lines[file][order].append([])           # add an empty list for each noise factor

                    line_QTable = find_lines_threshold(spectrum, noise_factor=nf_list[nf]) # Identify lines, generate QTable

                    lines[file][order][nf].append(line_QTable)# append the QTable

            elif order == 1 :                               # ... and Order 3 only!
                lines[file].append([])                      # add an empty list for each QTable

                noise_region = region_3                     # Point to the region to determine noise
                spectrum = spec1D_subbed[file][order][0]    # Point to the Spec1D object
                spectrum = noise_region_uncertainty(spectrum, noise_region) # Estimate uncertainty

                for nf in range(len(nf_list)):              # range through each noise factor
                    lines[file][order].append([])           # add an empty list for each noise factor

                    line_QTable = find_lines_threshold(spectrum, noise_factor=nf_list[nf]) # Identify lines, generate QTable

                    lines[file][order][nf].append(line_QTable)      # append the QTable

    return lines

def lineMatching(QTable_list, database, username, pword, table, colnames, datatype, unit_list, distance, rs):
    '''
    Function == Match the lines in each QTable to the master table from Kaplan et al. 2017
    Input    == Nested list of QTables, database name, username for database, password, name of master table
                in database, name of columns to be selected, dataypes of each selected column, relevant unit
                of each selected column, INSERT HERE, redshift of object
    Output   == Nested list of tables containing the requested quantities for each IDed feature
    '''

    if 'vac_lamb' not in colnames:                                          # check if the vacuum wavelengths have been requested
        print('!!! Vacuum wavlengths not requested: aborting lineMatching !!!')
        print("!!! Add 'vac_lamb' to columns under VARIABLES !!!")
        return                                                              # print error message and terminate function if not requested

    '''
    PART 1 || OBTAIN THE MASTER TABLE
    Part   == Connect to the Postgres database and copy the H_2 table from Kaplan et al. 2017
    Output == Table from the database with requested columns
    '''
    conn = psycopg2.connect(dbname=database, user=username, password=pword) # establish a connection
    cursor = conn.cursor()                                                  # create a cursor object

    query = col_append('SELECT ', len(colnames), colnames)                  # start your query
    query = query + ' FROM ' + table + ';'                                  # finish your query

    cursor.execute(query)                                                   # execute the query

    records = cursor.fetchall()                                             # point to the results of the query
    array   = np.array(records)                                             # turn into an array
    table   = Table(array, names=colnames, dtype=datatype)                  # turn into a table with columns named and typed
    table   = QTable(table)                                                 # let it accept quantities

    for unit in range(len(unit_list)):                                      # check the unit of each column
        if unit_list[unit] != 'N/A' :                                       # only change if the column needs a unit
            table[colnames[unit]] = table[colnames[unit]] * unit_list[unit] # multiply by the relevant unit

    '''
    PART 2 || COMPARE TO THE MASTER TABLE AND APPEND TO NEW TABLE
    Part   == Determine which astropy emission lines are close to the expected H_2 wavelengths
    Output == Nested list of tables with matched features
    '''

    t_list = emptyFITSList(QTable_list)                                     # create empty lists for each FITS file

    vac_lm_i = unit_list.index(u.um)                                        # get index of vacuum wavelength column before appending observed data
    nf_t_cols = table.colnames                                              # point to list of requested column names

    nf_t_cols.insert(vac_lm_i + 1, 'delta_lmb')                             # append the column name, datatype, and units for the wavelength difference after the vacuum wavelength
    datatype.insert(vac_lm_i + 1, float)
    unit_list.insert(vac_lm_i + 1, u.nm)

    nf_t_cols.insert(vac_lm_i + 1, 'smc_lamb')                              # append the column name, datatype, and units for the observed wavelength after the vacuum wavelength (before wavelength difference)
    datatype.insert(vac_lm_i + 1, float)
    unit_list.insert(vac_lm_i + 1, u.um)

    for file in range(len(QTable_list)):                                    # range through each FITS file
        for order in range(len(QTable_list[file])):                         # range through each spectral order
            t_list[file].append([])                                         # add an empty list for each Spectrum1D object
            for nf in range(len(QTable_list[file][order])):                 # range through each noise factor
                t_list[file][order].append([])                              # add an empty list for each noise factor

                t = QTable(names=nf_t_cols, dtype=datatype)                 # create a table with each column name and datatype

                smc_table = QTable_list[file][order][nf][0]                 # point to the nf QTable
                smc_center = smc_table[smc_table['line_type'] == 'emission']['line_center']
                # point to the IDed emission line centers
                for center in range(len(smc_center)):                       # range through each IDed emission line
                    lamb_center = smc_center[center].value                  # extract the value (remove the unit)
                    lamb_rest   = lamb_center - rs                          # subtract the redshift for home frame wavelength

                    lower_bound = lamb_rest - distance                      # get a lower bound
                    upper_bound = lamb_rest + distance                      # get an upper bound

                    match_list = list(filter(lambda x: lower_bound <= x <= upper_bound, table[colnames[vac_lm_i]].value))
                    # create a list of Kaplan IDs that are within the bounds of the IDed emission line
                    if bool(match_list):                                    # check if any lines were matched

                        if len(match_list) == 1 :                           # check if only one line was matched
                            lamb_theo = match_list[0]                       # point to the matched vacuum wavelength
                            delt_lamb = lamb_rest - lamb_theo               # calculate the wavelength difference

                        elif len(match_list) > 1 :                          # check if more than one line was matched
                            delt_lamb = []                                  # create a list to hold wavelength differences

                            for match_ID in range(len(match_list)):         # range through each matched ID
                                lamb_theo = match_list[match_ID]            # point to the matched ID
                                lamb_diff = lamb_rest - lamb_theo           # calculate the wavelength difference
                                delt_lamb.append(lamb_diff)                 # append the difference

                            calc = len(delt_lamb)                           # get the amount of wavelength differences calculated

                            while calc > 1:                                 # trim the matches and differences before appending
                                abso_delt = [abs(ele) for ele in delt_lamb] # get the absolute value of the wavelength differences
                                abso_inde = abso_delt.index(max(abso_delt)) # get the index of the largest difference
                                match_list.remove(match_list[abso_inde])    # remove the match that resulted in the largest difference
                                delt_lamb.remove(delt_lamb[abso_inde])      # remove the largest difference
                                calc = calc - 1                             # loop until one difference remains

                            lamb_theo = match_list[0]                       # point to the matched vacuum wavelength
                            delt_lamb = delt_lamb[0]                        # point to the smallest wavelength difference

                        table_vals = []                                     # create a list to hold the Kaplan table values

                        for colname in range(len(nf_t_cols)):               # range through each column
                            if nf_t_cols[colname] == 'vac_lamb':            # act on the vacuum wavelength column
                                col = nf_t_cols[colname]                    # point to the column
                                index = list(table['vac_lamb'].value).index(lamb_theo)
                                # find the index of the Kaplan table with the same vacuum wavelength
                                table_val = table[col][index]               # point to the vacuum wavelength from the Kaplan table

                                table_vals.append(table_val)                # append the vacuum wavelength
                                table_vals.append(lamb_rest)                # append the SMC wavelength
                                table_vals.append(round(delt_lamb * 10**3, 2))
                                # append the wavelength difference in units of 10 ** -3 u.um
                            elif nf_t_cols[colname] != 'delta_lmb' and nf_t_cols[colname] != 'smc_lamb':
                                # act on all other requested columns
                                col = nf_t_cols[colname]                    # point to the column
                                index = list(table['vac_lamb'].value).index(lamb_theo)
                                # find the index of the vacuum wavelength
                                table_val = table[col][index]               # point to to quantity in the Kaplan table

                                table_vals.append(table_val)                # append the quantity

                        row = table_vals                                    # point to the filled row
                        t.add_row(row)                                      # append the row

                for unit in range(len(unit_list)):                                      # check the unit of each column
                    if unit_list[unit] != 'N/A' :                                       # only change if the column needs a unit
                        t[t.colnames[unit]] = t[t.colnames[unit]] * unit_list[unit]     # multiply by the relevant unit
                t_list[file][order][nf].append(t)                                       # append the table to the returned list

    return t_list

def col_append(s, length, columns):
    '''
    Function == Add each desired column name to the PostgreSQL query
    Input    == Start of query, amount of columns, list of column names
    Output   == Start of query with appended column names (NOT the entire query)
    Modified == https://tinyurl.com/y4tcpwax, Author == Pankaj
    '''
    output = s                              # point to the starting query
    col = 0                                 # count how many columns you've ran through
    while col < length - 1:                 # work on all but the last column name
        output += columns[col] + ', '       # append the comma so more names can follow
        col = col + 1
    output += columns[col]                  # don't append a comma to the last column name
    return output

def normalFlux(tables, spectra, rs, rp):
    '''
    Function == Fit a normal to each matched feature and integrate the flux
    Input    == Nested list of tables, nested list of continuum subbed Spec1D objects, redshift, resolving power
    Output   == Nested list of tables with appended flux integrals and error
    '''
    for file in range(len(tables)):                                         # range through each FITS file
         for order in range(len(tables[file])):                             # range through each spectral order
             spectrum = spectra[file][order][0]                             # point to the order spectrum
             spectral_ax = spectrum.spectral_axis                           # pull out the spectral and flux axes
             flux_ax = spectrum.flux
             for nf in range(len(tables[file][order])):                     # range through each noise factor
                 table = tables[file][order][nf][0]                         # point to the table to be worked on
                 table['res'] = 0.0                                         # populate a res, err, and S/N column with floats
                 table['err'] = 0.0
                 table['S/N'] = 0.0
                 centers = table['smc_lamb'].value + rs                     # extract the IDed lines and apply the redshift
                 for center in range(len(centers)):                         # range through each IDed line
                    index = list(spectral_ax.value).index(centers[center]) # obtain the index of the IDed line in the spectrum
                    line  = spectrum[index-20:index+21]                    # extract the spectrum 20 points to the right and left of the line center
                    line_spectral_ax = line.spectral_axis                  # pull out the spectral and flux axes
                    line_flux_ax = line.flux

                    g_init = models.Gaussian1D(amplitude=findMiddle(spectrum, index, line_flux_ax), mean=findMiddle(spectrum, index, line_spectral_ax), stddev=rp*u.um)
                    # make an initial guess for the normal fit using the line center
                    g_fit = fit_lines(line, g_init)                        # fit the normal

                    index = list(line_spectral_ax.value).index(centers[center])
                    # point to the SQL table index
                    res, err, sn = 0.0, 0.0, 0.0                           # create dummy floats for the inegral and noise
                    res, err, sn = askFit(line_spectral_ax, line_flux_ax, g_fit, res, err, table['id'][center], index)
                    # assess validity of fit

                    table['res'][center] = res                              # append integrated sum and error
                    table['err'][center] = err
                    table['S/N'][center] = sn

                 table.remove_rows(table['S/N'] == 0.0)                     # remove rows with unsuccessful fits
                 table.remove_column('smc_lamb')                            # remove SMC column to resemble Kaplan table

    return tables

def findMiddle(spectrum_obj, point, input_list):
    '''
    Function == Find the mean of a list if it has an even length
    Input    == List
    Output   == Mean of list or error message
    '''
    middle = float(len(input_list))/2
    if middle % 2 != 0:
        return input_list[int(middle - .5)]
    else:
        print("Mean not on spectral axis -- array is odd -- adding one data point")
        length = len(input_list) + 1
        line = spectrum_obj[point-length:point+length+1]
        line_flux_ax = line.flux
        middle = float(len(line_flux_ax))/2
        return middle

def askFit(spectral, flux, model, integral, error, name, center):
    '''
    Function == Determine, by eye or through automation, whether or not the gaussian fit was successful
    Input    == line spectral axis, line flux axis, gaussian model, integrated sum, computed error, ID, center index
    Output   == line flux, line flux error, signal-to-noise ratio
    '''
    fig = plt.figure(figsize=(4,3))
    plt.plot(spectral.value, flux.value, 'b-')                              # plot the line
    plt.plot(spectral, model(spectral), color='magenta', linestyle='dashed')
    # plot the fit
    plt.plot(spectral.value[center], flux.value[center], 'b*', markersize = 15)
    # put a star on the centroid
    plt.title(name)                                                         # label the line

    flux_list = flux.value                                                  # pull out the values of the flux
    badIndices = statistics.mean(flux_list) > flux_list                     # create a mask of values greater than the flux mean
    plt.plot(spectral.value[badIndices], flux.value[badIndices], 'o--', color='grey', alpha=0.3)
    # plot the points considered to be continuum noise

    dev = statistics.stdev(flux_list[badIndices])                           # calculate the standard deviation of the continuum
    model_unitless = models.Gaussian1D(model.amplitude.value, model.mean.value, model.stddev.value)
    # create a copy of the 1D Gaussian model without units
    integral, error = quad(model_unitless, spectral[0].value, spectral[-1].value)
    # measure the line flux
    model_spectrum = Spectrum1D(flux=model(spectral), spectral_axis=spectral)
    # create a Spectrum1D object with the 1D Gaussian model
    width = gaussian_fwhm(model_spectrum)                                   # measure the FWHM of the line using the model
    error = width.value * dev                                               # calculate the noise for the line flux esitmate
    ratio = integral / error                                                # calculate the S/N ratio
    ratio = round(ratio, 1)                                                 # round off the S/N ratio

    print("=========================================== " + name)            # create a hash delimiter and print the results
    print("Line is                                   : % s"
        % (name))
    print("Standard Deviation of line is             : % s"
        % (dev))
    print("Line flux is                              : % s"
        % (integral))
    print("FWHM of line is                           : % s"
        % (width))
    print("Noise of line is                          : % s"
        % (error))
    print("S/N of line is                            : % s"
        % (ratio))
    print("===========================================")                    # close the hash delimiter

    '''
    USER INPUT
    '''
    '''
    while True:                                                             # create environment for user input
        test4num = input("Is fit good? (y/n) ")
        if test4num == 'y' :                                                # if the fit is good, break the loop
            break
        elif test4num == 'n' :                                              # if the fit is bad...
            integral = error = ratio = 0.0                                   # ... set the integral, error, and S/N to 0
            break
        else:
            print("Error! This is not a 'y' or 'n'. Try again.")            # check for desried inputs
    '''

    '''
    AUTOMATED
    '''
    #'''
    if (ratio >= 3.0 and ratio <= 100):                                 # if the fit is good, keep the measurements
        print('Very nice!')
    elif (ratio < 3) or (math.isnan(ratio)):                            # if the fit is bad...
        inegral = error = ratio = 0.0                                   # ... set the integral, error, and S/N to 0
    #'''

    plt.close('all')                                                        # close the plot
    return integral, error, ratio

def columnDensity(tables):
    '''
    Function == Calculate the plot measurments and normalize the flux of each line
    Input    == Nested list of tables
    Output   == List nested by nf of excitation energy and colDen/stat weight for each vib fam, modified tables
    '''
    colDens = emptyFITSList(tables)                         # create empty lists for each FITS file
    h = 6.626070e-27                                        # Planck's constant in cm^2*g*s^-1
    c = 2.997920e10                                         # The speed of light in cm*s^-1
    for file in range(len(colDens)):                        # range through each FITS file
        nf_list_A = tables[file][0]                         # point to order4 lists
        nf_list_B = tables[file][1]                         # point to order3 lists

        res_ref = [0.0, 0.0]                                # reference line flux
        res_err = [0.0, 0.0]                                # reference line error

        for nf in range(len(nf_list_A)):                    # range through each noise factor
            colDens[file].append([])                        # append an empty list for each noise factor
            orderA = nf_list_A[nf][0]                       # point to order4
            orderB = nf_list_B[nf][0]                       # point to order3
            stack = vstack([orderA, orderB])                # stack the orders into a single QTable
            plot_list = []                                  # create empty list for plottable measurements
            v_key = 1                                       # minimum vibrational family
            norm = [0.0, 0.0]                               # dummy float vars to hold the normalization line norm flux
            while (v_key <= 10):                            # pull out up to maximum vibrational family
                v = v_key
                xp = ([])                                   # empty lists for para data
                yp = ([])
                njp = ([])
                rotp = ([])
                xo = ([])                                   # empty lists for ortho data
                yo = ([])
                njo = ([])
                roto = ([])
                info = ([])                                 # empty list to hold everything
                for line in range(len(stack)):              # range through each line
                    vib_fam, rot_val = stack['v_u'][line], stack['j_u'][line]
                    # point to the vibrational family and the upper rotational value
                    if (vib_fam == v):                      # check if the line is in the vibrational family
                        euk, flux, aul, wave  = [float(stack['euk'][line].value), float(stack['res'][line]),
                            10 ** float(stack['aul'][line].value), float(stack['vac_lamb'][line].value) * 1000]
                        # point to the energy, the flux, the aul and undo the log, and the wavelength in cm
                        flux = flux * wave                  # correct the flux units
                        nj = flux / ( (1 / wave) * h * c * aul)
                        # calculate the column density of H2 in the upper J energy state
                        if (rot_val % 2 == 0):              # Checks for para
                            g = 2 * rot_val + 1             # calculate quantum degeneracy
                            xp.append(euk)
                            yp.append(nj/g)               # append statistically weighted colDen
                            njp.append(nj)
                            rotp.append(rot_val)
                        else:                               # checks for ortho
                            g = 3 * (2 * rot_val + 1)
                            xo.append(euk)
                            yo.append(nj/g)
                            njo.append(nj)
                            roto.append(rot_val)
                        if (vib_fam == 4) and (rot_val == 1):
                            # pull out first norm line option
                            norm[0] = nj/g

                            table = nf_list_B[2][0]                         # point to the table with norm line flux
                            mask = table['id'] == '4-2 O(3)'                # mask all but reference
                            res_ref[0] = table[mask]['res']
                            res_err[0] = table[mask]['err']
                        if (vib_fam == 1) and (rot_val == 3):
                            # pull out second norm line option
                            norm[1] = nj/g

                            table = nf_list_A[2][0]                         # point to the table with norm line flux
                            mask = table['id'] == '1-0 S(1)'                # mask all but reference
                            res_ref[1] = table[mask]['res']
                            res_err[1] = table[mask]['err']
                info.append(v)                              # append the vibrational family
                p = ([xp, yp, njp, rotp])                   # create para and ortho lists
                o = ([xo, yo, njo, roto])
                info.append(p)                              # append the lists
                info.append(o)
                plot_list.append(info)                      # append the information to the plot list
                v_key += 1                                  # move to the next vibrational family

            plot_list.append(norm)                          # append the 4-2 O(3) and 1-O S(1) column density
            colDens[file][nf].append(plot_list)

        ### MODIFYING FLUX IN TABLE ###
        for order in range(len(tables[file])):              # range through each order
            for nf in range(len(tables[file][order])):      # range through each NF
                table = tables[file][order][nf][0]          # point to the table to work on
                if (res_ref[0] != 0.0):                     # set flux and error to 4-2 O(3)...
                    res = res_ref[0]
                    err = res_err[0]
                else:                                       # ... or 1-0 S(1)
                    res = res_ref[1]
                    err = res_err[1]

                i_err = table['err'] / table['res']         # get line and reference line error
                r_err = err / res

                z = table['res'] / res                      # get normalized flux
                z_err = z * np.sqrt(i_err**2 + r_err**2)    # calc error

                table['res'] = np.log10(z)                  # get log10 of normalized flux
                table['err'] = table['res'] * 0.4343 * np.sqrt(i_err**2 + r_err**2)
                # calc error

                for line in range(len(table['res'])):       # range through each line and round to 3 places
                    table['res'][line] = round(table['res'][line], 3)
                    table['err'][line] = round(table['err'][line], 3)

    return colDens, tables

def readH2COLfile(models_list):
        '''
        Function == Read the H2col of each initial model and extract the column densities
        Input    == List of H2COL filenames
        Output   == Nested list of plottable measurments with 4-2 O(3) and 1-O S(1) line flux for model
        '''

        data = ([])

        for file in range(len(models_list)):                # range through each model
            f = open(models_list[file], "r")
            lines = f.readlines()                           # create a list using each line

            v_col = []                                      # create an empty list for each desired column
            j_col = []
            ener_col = []
            colden_col = []
            coldenw8_col = []

            lines.pop(0)                                    # remove the file header

            maxV = 10                                       # set max vib fam to be extracted
            for line in lines:                              # range through each line and append the data
                columns = line.split("\t")
                if int(columns[0]) <= maxV and int(columns[0]) != 0:
                    # grab data for vibrational families 1 through maxV
                    v_col.append(columns[0])
                    j_col.append(columns[1])
                    ener_col.append(columns[2])
                    colden_col.append(columns[3])
                    coldenw8_col.append(columns[4])

            f.close()                                       # close the file

            plot_list = []                                  # create an empty list for plottable measurements
            v_key = 1                                       # start with the first vib fam
            norm = [0.0, 0.0]                               # dummy floats to hold norm fluxes

            while (v_key <= maxV):
                v = v_key
                xp = ([])                                   # empty list for para...
                yp = ([])
                colp = ([])
                rotp = ([])
                xo = ([])                                   # and ortho energy and column density
                yo = ([])
                colo = ([])
                roto = ([])
                info = ([])                                 # list to hold all plot information
                for i in range(len(v_col)):
                    if (int(v_col[i]) == v):                # work on vib fam in key
                        if (int(j_col[i]) % 2 == 0):        # Checks for para
                            xp.append(float(ener_col[i]))
                            yp.append(float(coldenw8_col[i]))
                            colp.append(float(colden_col[i]))
                            rotp.append(int(j_col[i]))
                        else:
                            xo.append(float(ener_col[i]))
                            yo.append(float(coldenw8_col[i]))
                            colo.append(float(colden_col[i]))
                            roto.append(int(j_col[i]))
                    if (int(v_col[i]) == 4) and (int(j_col[i]) == 1):
                        # pull out 4-2 O(3) column density and line flux + error
                        norm[0] = float(coldenw8_col[i])
                    if (int(v_col[i]) == 1) and (int(j_col[i]) == 3):
                        # pull out 1-0 S(1) column density and line flux + error
                        norm[1] = float(coldenw8_col[i])

                info.append(v)
                p = ([xp, yp, colp, rotp])
                o = ([xo, yo, colo, roto])
                info.append(p)
                info.append(o)
                plot_list.append(info)                      # append the vib fam, para and ortho data
                v_key += 1
                if (v_key == 10 + 1):
                    # if all desired vibrational families have been appended...
                    plot_list.append(norm)                  # append the 4-2 O(3) and 1-O S(1) line flux

            data.append(plot_list)
        return data

def plotNratioNchi(tables, data, models):
    for model in range(len(models)):
        model_list = models[0]
        cld_nrm   = model_list[-1]
        for file in range(len(tables)):
            for order in range(len(tables[file])):
                for nf in range(len(tables[file][order])):
                    data_list = data[file][order][nf]

                    norm_list = data_list[-1]                           # point to the observed norm list
                    if (norm_list[0] != 0.0):                           # check if 4-2 O(3) is norm line
                        normO = norm_list[0]
                        normC = cld_nrm[0]
                    else:                                               # check if 1-O S(1)
                        normO = norm_list[1]
                        normC = cld_nrm[1]

                    model_list_copy = []
                    data_list_copy  = []

                    table_list = tables[file][order][nf]
                    table      = table_list[0]
                    #t          = Table() # empty table
                    #table_list.append(t)

                    for v in range(len(model_list) - 1):
                        if model_list[v][0] in table['v_u']:
                            model_list_copy.append(model_list[v])
                            data_list_copy.append(data_list[v])

                    for line in range(len(table)):
                        for v in range(len(model_list_copy)):
                            if table[line]['v_u'] == model_list_copy[v][0]:
                                load_m = model_list_copy[v]
                                load_d = data_list_copy[v]

                        load_m_euk = load_m[1][0] + load_m[2][0]
                        load_d_euk = load_d[1][0] + load_d[2][0]

                        load_m_euk_round = [round(euk, 0) for euk in load_m_euk]

                        load_m_coldenw8 = load_m[1][1] + load_m[2][1]
                        load_d_coldenw8 = load_d[1][1] + load_d[2][1]

                        load_m_colden = load_m[1][2] + load_m[2][2]
                        load_d_colden = load_d[1][2] + load_d[2][2]


                        eukT = table[line]['euk'].value

                        maskm = eukT == load_m_euk_round
                        maskd = eukT == load_d_euk

                        coldenw8m = np.array(load_m_coldenw8)[maskm]
                        coldenw8d = np.array(load_d_coldenw8)[maskd]

                        coldenm = np.array(load_m_colden)[maskm]
                        coldend = np.array(load_d_colden)[maskd]

                        print("Line is                                       : % s"
                            % (table[line]['id']))
                        print("Normalized column density is                  : % s"
                            % (np.log(coldenw8d / normO)))


    return

'''
VARIABLES
'''
#! FITS_list = ([string of  ARCoIRIS filenames])
# example: (['lk250_xtc_ergs.fits', 'LK215_XTC.fits', 'lk210_xtc_ergs.fits'])

#! models_list = ([string of h2col filenames])
# example: (['kaplan_rec_noCMB.h2col', 'barria_rec_noOuter.h2col', 'ppd_11.h2col'])

region_order_4 = SpectralRegion(1.95*u.um, 2.43*u.um)
region_order_3 = SpectralRegion(1.45*u.um, 1.80*u.um)
# SpectralRegion objects to remove the telluric corruption of each relevant order in our data

#! noise_factor_trials = [noise factor for find_lines_threshold in lineFinding]
# example: [3, 2, 1]

#! dbname    = string of name of database
#! user      = string of username
#! password  = string of password
# connection parameters for psycopg2.connect()

sql_table = 'Kaplan'
columns   = ['vac_lamb', 'id', 'v_u', 'j_u', 'euk', 'aul']
dtype     = [float, str, int, int, int, float]
units     = [u.um, 'N/A', 'N/A', 'N/A', u.K, u.s ** -1]
# information about the desired columns from the PostgreSQL table
# columns, dtype, and units MUST have matching indices !!!

delta = 0.0007
# resolving power (R = 3000) for the ARCoIRIS spectrograph

redshift = 0.0012
# redshift of source

'''
PROGRAM
'''
def main(FITS_list):
    data_list     = readFITSdata(FITS_list)
    trim_list     = trimNAN(data_list)      # might be memory intensive for many FITS files

    spectrum_list = spec1D(trim_list)
    telu_list     = teluBgone(spectrum_list, region_order_4, region_order_3)
    subbed_list   = fitNsubList(telu_list)

    lines_list    = lineFinding(subbed_list, region_order_4, region_order_3, noise_factor_trials)
    table_list    = lineMatching(lines_list, dbname, user, password, sql_table, columns, dtype, units, delta, redshift)
    table_list    = normalFlux(table_list, subbed_list, redshift, delta)
    plot_list, table_list     = columnDensity(table_list)

    clod_list     = readH2COLfile(models_list)

    #plotNratioNchi(table_list, plot_list, clod_list)

    return spectrum_list, subbed_list, lines_list, table_list, plot_list, clod_list

'''
RUN PROGRAM
'''
unsubbed, subbed, lines, tables, data, models = main(FITS_list)

'''
PLOTTING AND POST-ANALYSIS
Draft: 03/03/20
'''
colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
#From Color Universal Design (CUD): https://jfly.uni-koeln.de/color/

for file in range(len(data)):                               # range through each FITS file
    plot_list = data[file][2][0]                            # point to the line flux data
    norm_list = plot_list[-1]                               # point to the observed norm list
    table = vstack([tables[file][1][2][0],tables[file][0][2][0]]) # stack the tables for the orders in the file

    for h2col in range(len(models)):                        # range through each initial model
        model = models[h2col]                               # point to the extracted data
        cld_nrm = model[-1]                                 # point to the Cloudy norm list

        plt.figure(figsize=(8.4, 4.8))

        if (norm_list[0] != 0.0):                           # check if 4-2 O(3) is norm line
            normd = norm_list[0]
            normm = cld_nrm[0]
        else:                                               # check if 1-O Q(4)
            normd = norm_list[1]
            normm = cld_nrm[1]
        for v in range(len(plot_list) - 1):                 # range through each vib fam
            para_xd = plot_list[v][1][0]                    # point to the Cloudy para data
            para_yd = plot_list[v][1][1]
            ortho_xd = plot_list[v][2][0]                   # point to the Cloudy ortho data
            ortho_yd = plot_list[v][2][1]
            plt.plot(para_xd, np.log([x / normd for x in para_yd]), 'o',  color = colors[v % 8])
            # plot the observed para data
            plt.plot(ortho_xd, np.log([x / normd for x in ortho_yd]), '^', color = colors[v % 8])
            # plot the observed ortho data
            if (len(plot_list[v][1][0]) == 0) and (len(plot_list[v][2][0]) == 0):
                print('No data for v = ' + str(v + 1) + ' , did not plot model')
            else:
                para_xm = model[v][1][0]   # point to the Cloudy para data
                para_ym = model[v][1][1]
                ortho_xm = model[v][2][0]  # point to the Cloudy ortho data
                ortho_ym = model[v][2][1]
                plt.plot(para_xm, np.log([x / normm for x in para_ym]), color = colors[v % 8], linestyle = '--')
                # plot the Cloudy para data
                plt.plot(ortho_xm, np.log([x / normm for x in ortho_ym]), color = colors[v % 8], linestyle = '-')
                # plot the Cloudy ortho data
        plt.ylabel('Column Density ' + r'$\ln{\frac{N_J}{g_J}} - \ln{\frac{N_r}{g_r}}$')
        plt.xlabel('Excitation Energy ' + r'$\left( \frac{E_J}{k} \right)$' + ' (K)')
        print("Plotted " + FITS_list[file][:5] + " and overlaid " + models_list[h2col][:6])
        print("==================================================")
        # print which point source and which model were plotted

        model_list_copy = []
        plot_list_copy  = []

        for v in range(len(model) - 1):
            if model[v][0] in table['v_u']:
                model_list_copy.append(model[v])
                plot_list_copy.append(plot_list[v])

        colA = []
        colB = []
        for line in range(len(table)):
            for v in range(len(model_list_copy)):
                if table[line]['v_u'] == model_list_copy[v][0]:
                    load_m = model_list_copy[v]
                    load_d = plot_list_copy[v]

            load_m_euk = load_m[2][0] + load_m[1][0]
            load_d_euk = load_d[2][0] + load_d[1][0]

            load_m_euk_round = [round(euk, 0) for euk in load_m_euk]

            load_m_coldenw8 = load_m[2][1] + load_m[1][1]
            load_d_coldenw8 = load_d[2][1] + load_d[1][1]

            load_m_colden = load_m[2][2] + load_m[1][2]
            load_d_colden = load_d[2][2] + load_d[1][2]

            load_m_rot = load_m[2][3] + load_m[1][3]
            load_d_rot = load_d[2][3] + load_d[1][3]

            eukT = table[line]['euk'].value     ### SHOULD I USE THE ID AS A KEY?

            maskm = eukT == load_m_euk_round
            maskd = eukT == load_d_euk

            coldenw8m = np.array(load_m_coldenw8)[maskm]
            coldenw8d = np.array(load_d_coldenw8)[maskd]

            coldenm = np.array(load_m_colden)[maskm]
            coldend = np.array(load_d_colden)[maskd]

            rotm = np.array(load_m_rot)[maskm]
            rotd = np.array(load_d_rot)[maskd]

            print("Line is                                       : % s"
                % (table[line]['id']))
            print("Normalized column density is                  : % s"
                % (np.log(coldenw8d / normd)))
            print("Energy is                                     : % s"
                % (eukT))
            print("V and J are                                   : % s, % s"
                % (table[line]['v_u'], rotd))
        print(" ")

# plotting with multiple axes
for file in range(len(data)):                               # range through each FITS file
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8.4, 4.8))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    for h2col in range(len(models)):                        # range through each initial model
        model = models[h2col]                               # point to the extracted data
        cld_nrm = model[-1]                                 # point to the Cloudy norm list

        plot_list = data[file][2][0]                        # point to the line flux data
        norm_list = plot_list[-1]                           # point to the observed norm list

        if (norm_list[0] != 0.0):                           # check if 4-2 O(3) is norm line
            normO = norm_list[0]
            normC = cld_nrm[0]
        else:                                               # check if 1-O Q(4)
            normO = norm_list[1]
            normC = cld_nrm[1]
        if (h2col == 0):
            axe = ax1
        elif (h2col == 1):
            axe = ax2
            #axe.set_xlabel('Excitation Energy ' + r'$\left( \frac{E_J}{k} \right)$' + ' (K)')
        for v in range(len(plot_list) - 1):                 # range through each vib fam
            axe.plot(plot_list[v][1][0], np.log(plot_list[v][1][1]) - np.log(normO), 'o',  color = colors[v % 8])
            # plot the observed para data
            axe.plot(plot_list[v][2][0], np.log(plot_list[v][2][1]) - np.log(normO), '^', color = colors[v % 8])
            axe.set_xlim([5000, 40000])
            # plot the observed ortho data
            if (len(plot_list[v][1][0]) == 0) and (len(plot_list[v][2][0]) == 0):
                print('No data for v = ' + str(v + 1) + ' , did not plot model')
            else:
                para_x = model[v][1][0]   # point to the Cloudy para data
                para_y = model[v][1][1]
                ortho_x = model[v][2][0]  # point to the Cloudy ortho data
                ortho_y = model[v][2][1]
                axe.plot(para_x, np.log(para_y) - np.log(normC), color = colors[v % 8], linestyle = '--')
                # plot the Cloudy para data
                axe.plot(ortho_x, np.log(ortho_y) - np.log(normC), color = colors[v % 8], linestyle = '-')
                # plot the Cloudy ortho data
        plt.ylabel('Column Density ' + r'$\ln{\frac{N_J}{g_J}}$', labelpad=12)
        plt.xlabel('Excitation Energy ' + r'$\left( \frac{E_J}{k} \right)$' + ' (K)', labelpad=12)
        print("Plotted " + FITS_list[file][:5] + " and overlaid " + models_list[h2col][:6])
        print(" ")
        # print which point source and which model were plotted

'''
JUNK (?)
'''

'''
patch01 = mpatches.Patch(color=colors[0], label='v=1')
patch02 = mpatches.Patch(color=colors[1], label='v=2')
patch04 = mpatches.Patch(color=colors[3], label='v=4')
patch05 = mpatches.Patch(color=colors[4], label='v=5')
patch06 = mpatches.Patch(color=colors[5], label='v=6')
patch07 = mpatches.Patch(color=colors[6], label='v=7')
black_dot = mlines.Line2D([0], [0], linewidth=0, markeredgecolor='black', marker='o',
    markersize=12, label='Para', fillstyle='none')
black_tri = mlines.Line2D([0], [0], linewidth=0, markeredgecolor='black', marker='^',
    markersize=12, label='Ortho', fillstyle='none')
for v in range(len(plot_list)):
    plt.plot(plot_list[v][1][0], np.log(plot_list[v][1][1]) - np.log(norm), 'o',  color = colors[v % 8])
    plt.plot(plot_list[v][2][0], np.log(plot_list[v][2][1]) - np.log(norm), '^', color = colors[v % 8])
plt.legend(handles=[patch01,patch02,patch04,patch05,patch06,patch07,black_dot,black_tri], loc=3.)
plt.ylabel('Column Density ' + r'$\ln{\frac{N_J}{g_J}}$')
plt.xlabel('Excitation Energy ' + r'$\left( \frac{E_J}{k} \right)$' + ' (K)')
'''
