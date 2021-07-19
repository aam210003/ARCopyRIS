# ARCopyRIS

ARCopyRIS is a Python script that processes TripleSpec4.1 ARCoIRIS (Astronomy Research using the Cornell Infra Red Imaging Spectrograph) data. The script removes telluric noise, fits and subtracts the continuum, and integrates molecular hydrogen emission lines from Kaplan et al. (2017).

It uses the following Python modules:

  * numpy
  * matplotlib
  * astropy
  * specutils
  * psycopg2
  * scipy
  * statistics
  * math

ARCopyRIS returns the unsubbed spectra, subbed spectra, and recreated Kaplan table for each file it processes.
