# Switching_EIT_Package
## Introduction
This package is used for calculating HOT ATOM Switching EIT.\
There are two classes in this package, Theory and ReadData.
## Classes
### Theory ( Omega_c, shift, Omega_s, OD, gamma_2, T )
Calculate the transparency of the probe laser which detuned between -40MHz to 40MHz.\
The definition of  arguments are as belowing:
* Omega_c : Rabifrequency of the Coupling laser ( The input value will be multiplied by Gamma(2XpiX5.75MHz)
* shift : The frequency detune of Coupling laser
* Omega_s : Rabifrequency of the Switching laser
* OD : The optical depth of the atoms
* gamma_2 : Dephasing rate of the system
* T : Temperature of the atoms, Unit : Celcius
#### Data(self)
This function will return two arrays Probe detuning and Transparency.
* Probe detuning : Unit : MHz
* Transparency : values from 0 to 1
#### FittingData(Probe Detuning Data)
