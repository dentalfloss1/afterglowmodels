import numpy as np 
import configparser
import argparse
import matplotlib.pyplot as plt
import datetime 
import sys
sys.path.append('../')
import vdH_equations as eqn 
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
    nuc_val = eqn.nuc(epsilonB, n0, E, D_L, z, t_sec)
    num_val = eqn.num(epsilonB, epsilonE, n0, E, D_L, z, t_sec, p) 
    nua1_val = eqn.nua1(epsilonB, epsilonE, n0, E, D_L, z, t_sec, p)
    nua2_val = eqn.nua2(epsilonB, epsilonE, n0, E, D_L, z, t_sec)
    nua3_val = eqn.nua3(epsilonB, epsilonE, n0, E, D_L, z, t_sec, p)
    spectrum5_indices = (nua2_val < nuc_val) & (nuc_val < num_val)
    
    spectrum1_indices = (nua1_val < num_val) & (num_val < nuc_val) 
    
    spectrum2_indices = (num_val < nua3_val) & (nua3_val < nuc_val)
    
    # spectrum2_indices = ~np.zeros(spectrum5_indices.shape, dtype=bool)
    # spectrum1_indices= np.zeros(spectrum5_indices.shape, dtype=bool)
    spec5segmentCcond = (nu < nua2_val) & spectrum5_indices
    spec5segmentEcond = (nu > nua2_val) & (nu < nuc_val) & spectrum5_indices
    spec5segmentFcond = (nu > nuc_val) & (nu < num_val) & spectrum5_indices
    spec5segmentHcond = (nu > num_val) & spectrum5_indices
    
    spec1segmentBcond = (nu < nua1_val) & spectrum1_indices
    spec1segmentDcond = (nu > nua1_val) & (nu < num_val) & spectrum1_indices
    spec1segmentGcond = (nu > num_val) & (nu < nuc_val) & spectrum1_indices
    spec1segmentHcond = (nu > nuc_val) & spectrum1_indices
    
    spec2segmentBcond = (nu < num_val) & spectrum2_indices
    # spec2segmentAcond = (nu > nu_m4_val)  & spectrum2_indices
    spec2segmentAcond = (nu > num_val) & (nu < nua3_val) & spectrum2_indices
    # spec2segmentGcond = (nu > nu_c3_val) & spectrum2_indices
    spec2segmentGcond = (nu > nua3_val) & (nu < nuc_val) & spectrum2_indices
    spec2segmentHcond = (nu > nuc_val) & spectrum2_indices
    
    condlist = [spec5segmentCcond, spec5segmentEcond, spec5segmentFcond, spec5segmentHcond, spec1segmentBcond, spec1segmentDcond, spec1segmentGcond, spec1segmentHcond, spec2segmentBcond, spec2segmentAcond, spec2segmentGcond, spec2segmentHcond]
    condnamelist = ['5C', '5E', '5F','5H', '1B', '1D', '1G', '1H' , '2B', '2A' ,'2G', '2H']
    transitionNU = []
    transitionName = []
    for c in condlist:
        if np.sum(c) != 0 :
            leftregion = np.sort(np.where(c==True)[0])[-1]
            print(leftregion)
            transitionNU.append(nu[leftregion])
    rgnatfreq = np.zeros(nu.shape, dtype='U2')
    print(transitionNU)
    for i in range(len(nu)):
        for j in range(len(condlist)):
            if condlist[j][i] == True:
                rgnatfreq[i] = condnamelist[j]
            else: 
                pass
    _, idx = np.unique(rgnatfreq[rgnatfreq != ''], return_index=True)
    uniquergn = rgnatfreq[rgnatfreq != ''][np.sort(idx)]
    print(uniquergn)
    for i in range(0,len(uniquergn)-1):
        transitionName.append(uniquergn[i]+"|"+uniquergn[i+1])
    # for n,t in zip(nu_m4_arr, t_days):
    #     print(t,'{:e}'.format(nu), '{:e}'.format(n))
    
    
    
    fluxes = np.zeros(nu.shape, dtype=float)
    
    
    fluxes[spec5segmentCcond] = eqn.F1( epsilonB, epsilonE, n0, E, D_L, z, t_sec, nu[spec5segmentCcond])
    fluxes[spec5segmentEcond] = eqn.F2( epsilonB, epsilonE, n0, E, D_L, z, t_sec, nu[spec5segmentEcond])
    fluxes[spec5segmentFcond] = eqn.F3( epsilonB, n0, E, D_L, z, t_sec, nu[spec5segmentFcond])
    fluxes[spec5segmentHcond] = eqn.F4(epsilonB, epsilonE, n0, E, D_L, z, t_sec ,p, nu[spec5segmentHcond])
    
    fluxes[spec1segmentBcond] = eqn.F5(epsilonB, epsilonE, n0, E, D_L, z, t_sec, p, nu[spec1segmentBcond])
    fluxes[spec1segmentDcond] = eqn.F6(epsilonB, epsilonE, n0, E, D_L, z, t_sec , p, nu[spec1segmentDcond])
    fluxes[spec1segmentGcond] = eqn.F7(epsilonB, epsilonE, n0, E, D_L, z, t_sec ,p, nu[spec1segmentGcond])
    fluxes[spec1segmentHcond] = eqn.F8(epsilonB, epsilonE, n0, E, D_L, z, t_sec ,p, nu[spec1segmentHcond])
    
    fluxes[spec2segmentBcond] = eqn.F9(epsilonB, epsilonE, n0, E, D_L, z, t_sec ,p, nu[spec2segmentBcond])
    fluxes[spec2segmentAcond] = eqn.F10(epsilonB, epsilonE, n0, E, D_L, z, t_sec ,p, nu[spec2segmentAcond])
    fluxes[spec2segmentGcond] = eqn.F11(epsilonB, epsilonE, n0, E, D_L, z, t_sec ,p, nu[spec2segmentGcond])
    fluxes[spec2segmentHcond] = eqn.F12(epsilonB, epsilonE, n0, E, D_L, z, t_sec ,p, nu[spec2segmentHcond])
    
    fluxes = np.copy(fluxes)*1e29
    
    
    fig = plt.figure
    plt.scatter(nu, fluxes, marker='o', s=0.1)# , color='lightsteelblue')
    
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(nu_min*0.5,nu_max*1.5)
    # ax.set_xlim(nu_min*0.5,nu_max*1.5)
    ax.set_ylim(1e-8, 1e5)
    # ax.set_ylim(np.amin(fluxes[fluxes!=0])*0.5, np.amax(fluxes)*1.5)
    ax.set_xlabel('nu (Hz)')
    ax.set_ylabel('F (μJy)')
    plt.title('Spectrum at '+str(t_days)+' days')
    
    ax.legend()
    plt.savefig('spectrum'+str(t_days)+'.png')
    plt.close()    
    if not isQuiet:
        for n,f in zip(nu, fluxes):
            print('{:e}'.format(n), '{:e}'.format(f))
    
    
    
    
    # PLOT SPECTRA AT GIVEN TIMES LOOK FOR MISSING SEGMENTS ? Maybe not?
    # Why is nu_m not crossing nu_obs properly ?


# GRB PARAMS
p = float(params['PARAMETERS']['p'])
z = float(params['PARAMETERS']['z'])
epsilonB = float(params['PARAMETERS']['epsilon_B'])
epsilonE = float(params['PARAMETERS']['epsilon_e'])
n0 = float(params['PARAMETERS']['n0'])
E_gamma53 = float(params['PARAMETERS']['E_gamma53'])
epsilon_gamma = float(params['PARAMETERS']['epsilon_gamma'])
t_days_min = float(params['PARAMETERS']['t_days_min'])
t_days_max = float(params['PARAMETERS']['t_days_max'])
t_days = float(params['PARAMETERS']['t_days'])
d_L28 = float(params['PARAMETERS']['d_L28'])
nu_min = float(params['PARAMETERS']['nu_min'])
nu_max = float(params['PARAMETERS']['nu_max'])
E_53 = E_gamma53*((1/epsilon_gamma) - 1)
    

# Computation params
res = int(float(params['PARAMETERS']['res']))
nu = np.geomspace(nu_min, nu_max, num=res)

t_sec = t_days*3600*24
D_L = d_L28*1e28
E = E_53*1e53

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
