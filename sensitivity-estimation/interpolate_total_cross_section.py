#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

interpolate total cross section in WW approx in units of cm2

Created on Thu Dec 14 17:09:20 2023

@author: axel
"""



import numpy as np
import scipy as sp
from scipy.interpolate import interp1d, interp2d
import matplotlib.pyplot as plt
import os
import pandas as pd

###########################
def plot_interpolation(df, interpfunc, mass):
    E0array = np.logspace(1,5,1000)
    totcsarray = [totcsInterpolation(energy) for energy in E0array]
    # plot
    plt.figure()
    plt.plot(df["E0"],df["totcs"],'k+', label="table entries")
    plt.plot(E0array,totcsarray,'b',label="interpolaton")
    plt.ylabel('$\sigma \; [cm^2]$')
    plt.xlabel('$E_0 \; [GeV]$')
    plt.legend()
    plt.grid()
    plt.xscale("log")
    plt.xlim([10,10000])
    plt.title(mass)
    plt.savefig("interpolation_plots/1D_interpolation_"+"{:.3f}".format(mass)+".png")
    plt.show()

def extract_mass(filename):
    """ totcs_WW_m_xyz.csv is format of files """
    index = filename.find("m_")
    return float(filename[index+2:-4])

###########################


datapath = "cross_section_tables/"
filenames = os.listdir(datapath)
masses = [extract_mass(filename) for filename in filenames]

# create interpolations and plot, both 1d and 2d interpolations
df_list = []
test_cs_2D = [] # for comparing 1D and 2D interpolation
E0=1000 # for comparing 1D and  2D interpolation
for filename in filenames:
    mass = extract_mass(filename)
    # read in from tables
    df = pd.read_csv(datapath+filename, names=["E0", "totcs"])
    df["mass"] = pd.Series([mass for i in range(len(df["E0"]))])
    
    # create interpolation
    totcsInterpolation = interp1d(df["E0"], df["totcs"],kind="quadratic")
    
    # plot
    plot_interpolation(df, totcsInterpolation, mass)
    
    # save for 2d interpolation
    test_cs_2D.append(totcsInterpolation(E0)) 
    df_list.append(df) 


# testing 2D interpolation
df_tot = pd.concat(df_list)
totcs_E_m_interpolation = interp2d(df_tot["E0"], df_tot["mass"], df_tot["totcs"],kind="cubic")

plt.figure()
plt.plot(masses, totcs_E_m_interpolation(E0, masses), "k+", label="2d interpolation")
plt.plot(masses, test_cs_2D, 'b.', label="1d interpolation")
plt.legend()
plt.title("Interpolation results for E0 =" + str(E0))
plt.ylabel("$\sigma \; [cm^2]$")
plt.xlabel("Mass [Gev]")
plt.savefig("1d_vs_2d_interpolation.png")
plt.show()