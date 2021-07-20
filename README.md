# ARCopyRIS

ARCopyRIS is a Python script that processes TripleSpec4.1 ARCoIRIS (Astronomy Research using the Cornell Infra Red Imaging Spectrograph) data. The script removes telluric noise, fits and subtracts the continuum, and integrates molecular hydrogen emission lines.

ARCopyRIS returns the unsubbed spectra, subbed spectra, and recreated Kaplan table for each file it processes. It can plot data from ARCoIRIS and the simulation code, Cloudy.

It uses the following Python modules that may not be auto installed with anaconda:

  * `astropy`
  * `specutils`
  * `psycopg2`

[_`astropy`_](https://docs.astropy.org/en/stable/install.html) and [_`specutils`_](https://specutils.readthedocs.io/en/stable/installation.html) handle FITS files, units, and spectra analysis. [_`psycopg2`_](https://www.psycopg.org/install/) interfaces with a PostgreSQL database. Click the hyperlink on each module for installation instructions.

## Instructions

Use `kaplan.csv` to create a Postgres table. Follow this [_link_](https://www.pgadmin.org/docs/pgadmin4/latest/index.html) to begin with pgAdmin 4, a GUI software for managing Postgres databases.

Once you have installed the Python modules and created a table with `kaplan.csv`, open the script and change all lines beginning with `#!` to the values relvant to your PC.

## Short Blurb

This script represents my undergraduate and post-bacc effots. In total, I spent about four years writing it. I am proud of the scope of this script, its rough but robust implementation, and the enthusiasm I maintained for the project as I continued to add to it. If you want to see my future projects that are (hopefully) more elegant through what I've learned, go to my [_personal webpage_](https://aamassoud.github.io/).
