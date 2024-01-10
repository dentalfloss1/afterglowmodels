import numpy as np 
import configparser
import argparse
import matplotlib.pyplot as plt
import datetime 
import sys
sys.path.append('../')
import GS_equations as eqn
import os 
import imageio 
from tqdm import tqdm
argparser = argparse.ArgumentParser()
argparser.add_argument("--animate", action='store_true', help="Make an movie of the evolution of the spectrum")
args = argparser.parse_args()

doAnimate = args.animate
params = configparser.ConfigParser()
params.read('parameters.config')

def maincalc(t_days, isQuiet):
    nu_sa1_val = eqn.nu_sa1(p, z, epsilon_e, epsilon_B, n0, E_52)
    nu_m2_val = eqn.nu_m2(p, z, epsilon_e, epsilon_B, E_52, t_days)
    nu_c3_val = eqn.nu_c3(p, z, epsilon_B, n0, E_52, t_days)
    nu_m4_val = eqn.nu_m4(p, z, epsilon_e, epsilon_B, E_52, t_days)
    nu_sa5_val = eqn.nu_sa5(p, z, epsilon_e, epsilon_B, n0, E_52, t_days)
    nu_sa6_val = eqn.nu_sa6(p, z, epsilon_e, epsilon_B, n0, E_52, t_days)
    nu_ac7_val = eqn.nu_ac7(p, z, epsilon_e, epsilon_B, n0, E_52, t_days)
    nu_sa8_val = eqn.nu_sa8(z, n0, E_52, t_days)
    nu_m9_val = eqn.nu_m9(p, z, epsilon_e, epsilon_B, E_52, t_days)
    nu_sa10_val = eqn.nu_sa10(z, epsilon_B, n0, E_52, t_days)
    nu_c11_val = eqn.nu_c11(z, epsilon_B, n0, E_52, t_days)
    if not isQuiet:
        print(nu_sa5_val)
        print(nu_m4_val)
    spectrum5_cond = (nu_ac7_val < nu_sa10_val) & (nu_sa10_val < nu_c11_val) & (nu_c11_val < nu_m9_val)
    spectrum5_cond_arr = np.full(nu.shape, spectrum5_cond)
    if not isQuiet:
        print("-------------------------")
        print("SPECTRUM 5")
        print(spectrum5_cond)
    
    
    spectrum1_cond = (nu_sa1_val < nu_m2_val) & (nu_m2_val < nu_c3_val) 
    spectrum1_cond_arr = np.full(nu.shape, spectrum1_cond)
    if not isQuiet:
        print("------------------------")
        print("SPECTRUM 1")
        print(spectrum1_cond)
    
    # spectrum2_cond = (nu_m4_val < nu_sa5_val) & (nu_sa5_val < nu_c3_val)
    spectrum2_cond =  (nu_sa5_val < nu_c3_val)
    spectrum2_cond_arr = np.full(nu.shape, spectrum2_cond)
    # spectrum2_cond = ~np.zeros(spectrum5_cond.shape, dtype=bool)
    # spectrum1_cond= np.zeros(spectrum5_cond.shape, dtype=bool)
    if not isQuiet:
        print("------------------------")
        print("SPECTRUM 2")
        print(spectrum2_cond)
        print((nu_m4_val < nu_sa5_val), (nu_sa5_val < nu_c3_val))
    spec5segmentBcond = (nu < nu_ac7_val) & spectrum5_cond_arr
    spec5segmentCcond = (nu > nu_ac7_val) & (nu < nu_sa10_val) & spectrum5_cond_arr
    spec5segmentEcond = (nu > nu_sa10_val) & (nu < nu_c11_val) & spectrum5_cond_arr
    spec5segmentFcond = (nu > nu_c11_val) & (nu < nu_m9_val) & spectrum5_cond_arr
    # spec5segmentFcond = (nu > nu_c11_val)  & spectrum5_cond_arr
    spec5segmentHcond = (nu > nu_m9_val) & spectrum5_cond_arr
    
    # print(spec5segmentBcond)
    # input("presskey")
    # print(spec5segmentCcond)
    # input("presskey")
    # print(spec5segmentEcond)
    # input("presskey")
    # print(spec5segmentFcond)
    # input("presskey")
    # print(spec5segmentHcond)
    # input("presskey")
    
    
    spec1segmentBcond = (nu <  nu_sa1_val) & spectrum1_cond_arr
    spec1segmentDcond = (nu > nu_sa1_val) & (nu < nu_m2_val) & spectrum1_cond_arr
    spec1segmentGcond = (nu > nu_m2_val) & (nu < nu_c3_val) & spectrum1_cond_arr
    spec1segmentHcond = (nu > nu_c3_val) & spectrum1_cond_arr
    
    
    
    spec2segmentBcond = (nu < nu_m4_val) & spectrum2_cond_arr
    spec2segmentAcond = (nu > nu_m4_val) & (nu < nu_sa5_val) & spectrum2_cond_arr
    spec2segmentGcond = (nu > nu_sa5_val) & (nu < nu_c3_val) & spectrum2_cond_arr
    spec2segmentHcond = (nu > nu_c3_val) & spectrum2_cond_arr
    
    
    
    fluxes = np.zeros(nu.shape, dtype=float)
    
    fluxes[spec5segmentBcond] = eqn.PLS_B(p, z, epsilon_e, n0, E_52, t_days, d_L28, nu_14[spec5segmentBcond] )
    fluxes[spec5segmentCcond] = eqn.PLS_C(p, z, epsilon_B, n0, E_52, t_days, d_L28, nu_14[spec5segmentCcond] )
    fluxes[spec5segmentEcond] = eqn.PLS_E(p, z, epsilon_B, n0, E_52, t_days, d_L28, nu_14[spec5segmentEcond] )
    fluxes[spec5segmentFcond] = eqn.PLS_F(p, z, epsilon_B, n0, E_52, t_days, d_L28, nu_14[spec5segmentFcond] )
    fluxes[spec5segmentHcond] = eqn.PLS_H(p, z, epsilon_e, epsilon_B, n0, E_52, t_days, d_L28, nu_14[spec5segmentHcond] )
    
    fluxes[spec1segmentBcond] = eqn.PLS_B(p, z, epsilon_e, n0, E_52, t_days, d_L28, nu_14[spec1segmentBcond] )
    fluxes[spec1segmentDcond] = eqn.PLS_D(p, z, epsilon_e, epsilon_B, n0, E_52, t_days, d_L28, nu_14[spec1segmentDcond] )
    fluxes[spec1segmentGcond] = eqn.PLS_G(p, z, epsilon_e, epsilon_B, n0, E_52, t_days, d_L28, nu_14[spec1segmentGcond] )
    fluxes[spec1segmentHcond] = eqn.PLS_H(p, z, epsilon_e, epsilon_B, n0, E_52, t_days, d_L28, nu_14[spec1segmentHcond] )
    
    fluxes[spec2segmentBcond] = eqn.PLS_B(p, z, epsilon_e, n0, E_52, t_days, d_L28, nu_14[spec2segmentBcond] )
    fluxes[spec2segmentAcond] = eqn.PLS_A(p, z, epsilon_B, n0, E_52, t_days, d_L28, nu_14[spec2segmentAcond] )
    fluxes[spec2segmentGcond] = eqn.PLS_G(p, z, epsilon_e, epsilon_B, n0, E_52, t_days, d_L28, nu_14[spec2segmentGcond] )
    fluxes[spec2segmentHcond] = eqn.PLS_H(p, z, epsilon_e, epsilon_B, n0, E_52, t_days, d_L28, nu_14[spec2segmentHcond] )
    
    
    fluxes = np.copy(fluxes)*1e3
    
    fig = plt.figure
    plt.scatter(nu, fluxes, marker='o', s=0.1, label='G&S2002')# , color='lightsteelblue')
    
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(nu_min*0.5,nu_max*1.5)
    ax.set_ylim(1e-8, 1e5)
    ax.set_xlabel('nu (Hz)')
    ax.set_ylabel('$F (μJy)$')
    plt.title('Spectrum at '+str(t_days)+' days')
    
    ax.legend()
    plt.savefig('spectrum'+str(t_days)+'.png')
    plt.close()    
    if not isQuiet:
        for n,f in zip(nu, fluxes):
            print('{:e}'.format(n), '{:e}'.format(f))
    
    
    
    


# GRB PARAMS
p = float(params['PARAMETERS']['p'])
z = float(params['PARAMETERS']['z'])
epsilon_B = float(params['PARAMETERS']['epsilon_B'])
epsilon_e = float(params['PARAMETERS']['epsilon_e'])
n0 = float(params['PARAMETERS']['n0'])
E_gamma53 = float(params['PARAMETERS']['E_gamma53'])
epsilon_gamma = float(params['PARAMETERS']['epsilon_gamma'])
t_days_min = float(params['PARAMETERS']['t_days_min'])
t_days_max = float(params['PARAMETERS']['t_days_max'])
t_days = float(params['PARAMETERS']['t_days'])
d_L28 = float(params['PARAMETERS']['d_L28'])
nu_min = float(params['PARAMETERS']['nu_min'])
nu_max = float(params['PARAMETERS']['nu_max'])
E_52 = E_gamma53*((1/epsilon_gamma) - 1)*1e-1
    
# Computation params
res = int(float(params['PARAMETERS']['res']))
nu = np.geomspace(nu_min, nu_max, num=res)


nu_14 = np.copy(nu)/1e14

if doAnimate:
    t_days_array = np.geomspace(t_days_min, t_days_max, num=res)
    isQuiet = True
    for t_days in tqdm(t_days_array):
        maincalc(t_days, isQuiet)
    with imageio.get_writer('animatedspectrum.gif', mode='I') as writer:
        for t_days in t_days_array:
            image = imageio.imread('spectrum'+str(t_days)+'.png')
            writer.append_data(image)
            os.remove('spectrum'+str(t_days)+'.png')
else:
    isQuiet = False
    maincalc(t_days, isQuiet)
