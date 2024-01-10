import numpy as np 
import configparser
import matplotlib.pyplot as plt
import datetime 
import sys 
sys.path.append('../')
import vdH_equations as eqn


params = configparser.ConfigParser()
params.read('parameters.config')

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
d_L28 = float(params['PARAMETERS']['d_L28'])
nu = float(params['PARAMETERS']['nu'])
nu_14 = nu/1e14

E_53 = E_gamma53*((1/epsilon_gamma) - 1)
# Computation params
res = int(float(params['PARAMETERS']['res']))
t_days = np.geomspace(t_days_min, t_days_max, num=res)


t_sec = t_days*3600*24
D_L = d_L28*1e28
E = E_53*1e53

# print('F_nuMax: ',np.amax(eqn.F_nuMax(epsilonB,n0,E,D_L,z,t_sec)*1e29),'uJy')

nuc_arr = eqn.nuc(epsilonB, n0, E, D_L, z, t_sec)
num_arr = eqn.num(epsilonB, epsilonE, n0, E, D_L, z, t_sec, p) 
nua1_arr = eqn.nua1(epsilonB, epsilonE, n0, E, D_L, z, t_sec, p)
nua2_arr = eqn.nua2(epsilonB, epsilonE, n0, E, D_L, z, t_sec)
nua3_arr = eqn.nua3(epsilonB, epsilonE, n0, E, D_L, z, t_sec, p)



spectrum5_indices = (nua2_arr < nuc_arr) & (nuc_arr < num_arr)

spectrum1_indices = (nua1_arr < num_arr) & (num_arr < nuc_arr) 

spectrum2_indices = (num_arr < nua3_arr) & (nua3_arr < nuc_arr)

# spectrum2_indices = ~np.zeros(spectrum5_indices.shape, dtype=bool)
# spectrum1_indices= np.zeros(spectrum5_indices.shape, dtype=bool)
spec5segmentCcond = (nu < nua2_arr) & spectrum5_indices
spec5segmentEcond = (nu > nua2_arr) & (nu < nuc_arr) & spectrum5_indices
spec5segmentFcond = (nu > nuc_arr) & (nu < num_arr) & spectrum5_indices
spec5segmentHcond = (nu > num_arr) & spectrum5_indices

spec1segmentBcond = (nu < nua1_arr) & spectrum1_indices
spec1segmentDcond = (nu > nua1_arr) & (nu < num_arr) & spectrum1_indices
spec1segmentGcond = (nu > num_arr) & (nu < nuc_arr) & spectrum1_indices
spec1segmentHcond = (nu > nuc_arr) & spectrum1_indices

spec2segmentBcond = (nu < num_arr) & spectrum2_indices
# spec2segmentAcond = (nu > nu_m4_arr)  & spectrum2_indices
spec2segmentAcond = (nu > num_arr) & (nu < nua3_arr) & spectrum2_indices
# spec2segmentGcond = (nu > nu_c3_arr) & spectrum2_indices
spec2segmentGcond = (nu > nua3_arr) & (nu < nuc_arr) & spectrum2_indices
spec2segmentHcond = (nu > nuc_arr) & spectrum2_indices



condlist = [spec5segmentCcond, spec5segmentEcond, spec5segmentFcond, spec5segmentHcond, spec1segmentBcond, spec1segmentDcond, spec1segmentGcond, spec1segmentHcond, spec2segmentBcond, spec2segmentAcond, spec2segmentGcond, spec2segmentHcond]
condnamelist = ['5C', '5E', '5F','5H', '1B', '1D', '1G', '1H' , '2B', '2A' ,'2G', '2H']
transitionT = []
transitionName = []
#for c,cname in zip(condlist, condnamelist):
#    print("----------------------")
#    print(cname)
#    print(np.sort(np.where(c==True)[0]))
#    input("presskey")
#    
#exit()
for c in condlist:
    if np.sum(c) != 0 :
        leftregion = np.sort(np.where(c==True)[0])[-1]
        print(leftregion)
        transitionT.append(t_days[leftregion])
rgnattime = np.zeros(t_days.shape, dtype='U2')
print(transitionT)
for i in range(len(t_days)):
    for j in range(len(condlist)):
        if condlist[j][i] == True:
            rgnattime[i] = condnamelist[j]
        else: 
            pass



_, idx = np.unique(rgnattime[rgnattime != ''], return_index=True)
uniquergn = rgnattime[rgnattime != ''][np.sort(idx)]
print(uniquergn)
for i in range(0,len(uniquergn)-1):
    transitionName.append(uniquergn[i]+"|"+uniquergn[i+1])
# for n,t in zip(nu_m4_arr, t_days):
#     print(t,'{:e}'.format(nu), '{:e}'.format(n))



fluxes = np.zeros(t_days.shape, dtype=float)


fluxes[spec5segmentCcond] = eqn.F1( epsilonB, epsilonE, n0, E, D_L, z, t_sec[spec5segmentCcond], nu)
fluxes[spec5segmentEcond] = eqn.F2( epsilonB, epsilonE, n0, E, D_L, z, t_sec[spec5segmentEcond], nu)
fluxes[spec5segmentFcond] = eqn.F3( epsilonB, n0, E, D_L, z, t_sec[spec5segmentFcond], nu)
fluxes[spec5segmentHcond] = eqn.F4(epsilonB, epsilonE, n0, E, D_L, z, t_sec[spec5segmentHcond] ,p, nu)

fluxes[spec1segmentBcond] = eqn.F5(epsilonB, epsilonE, n0, E, D_L, z, t_sec[spec1segmentBcond], p, nu)
fluxes[spec1segmentDcond] = eqn.F6(epsilonB, epsilonE, n0, E, D_L, z, t_sec[spec1segmentDcond] , p, nu)
fluxes[spec1segmentGcond] = eqn.F7(epsilonB, epsilonE, n0, E, D_L, z, t_sec[spec1segmentGcond] ,p, nu)
fluxes[spec1segmentHcond] = eqn.F8(epsilonB, epsilonE, n0, E, D_L, z, t_sec[spec1segmentHcond] ,p, nu)

fluxes[spec2segmentBcond] = eqn.F9(epsilonB, epsilonE, n0, E, D_L, z, t_sec[spec2segmentBcond] ,p, nu)
fluxes[spec2segmentAcond] = eqn.F10(epsilonB, epsilonE, n0, E, D_L, z, t_sec[spec2segmentAcond] ,p, nu)
fluxes[spec2segmentGcond] = eqn.F11(epsilonB, epsilonE, n0, E, D_L, z, t_sec[spec2segmentGcond] ,p, nu)
fluxes[spec2segmentHcond] = eqn.F12(epsilonB, epsilonE, n0, E, D_L, z, t_sec[spec2segmentHcond] ,p, nu)

fluxes = np.copy(fluxes)*1e29
# Load in boxfit lightcurve for comparison 
# boxfitdata = np.loadtxt('boxfitlightcurve.txt', delimiter=',',dtype={'names': ('i', 't', 'nu', 'F'), 'formats': (int, float, float, float)})

print('Tdays, nu, nua2, nua3, num, rgn, flux')
# for i in range(len(t_days)):
#     print("{:e}".format(t_days[i]), "{:e}".format(nu), "{:e}".format(nua2_arr[i]), "{:e}".format(nua3_arr[i]),  "{:e}".format(num_arr[i]),  rgnattime[i], fluxes[i], sep=',')

fig = plt.figure
plt.scatter(t_days, fluxes, marker='.',s=0.1, color='black')
# plt.scatter(boxfitdata['t']/3600/24, boxfitdata['F'], marker='*',  label='boxfit', color='darkorange')
wherefluxmax = np.argmax(fluxes)
print('Fmax: ',fluxes[wherefluxmax],' at t: ',t_days[wherefluxmax],' days')

print('F_nuMax: ',np.amax(eqn.F_nuMax(epsilonB,n0,E,D_L,z,t_sec[wherefluxmax])*1e29),'uJy')
# over90microJy = (fluxes > 90)
# print("Over 90 microJy between ",t_days[over90microJy].min(), "and", t_days[over90microJy].max(),"days.")
# print("Total days over 90 microJy: ",t_days[over90microJy].max()-t_days[over90microJy].min())

t1ind = np.array([round(t,1)==1 for t in t_days])

print('F: ',fluxes[t1ind],' at t: 1 days')
for myt in [0.3,2.2,4.2,8.3]:
    print('obsdate: ',myt)
    print(fluxes[np.round(t_days,decimals=1)==myt])
# from scipy.optimize import brentq

# def functosolve(t):
#     return eqn.nu_m4(p, z, epsilonE, epsilonB, E_52, t)-nu
# t_m4_days = brentq(functosolve, t_days_min, t_days_max)
# Fnu_m4_in = eqn.Fnu_m4(p, z, epsilonE, epsilonB, n0, E_52, t_m4_days, d_L28)

# plt.scatter(t_m4_days, Fnu_m4_in, marker='x', label='Fnu_m4')

# def functosolve2(t):
#     return eqn.nu_m2(p, z, epsilonE, epsilonB, E_52, t) - nu
# t_m2_days = brentq(functosolve2, t_days_min, t_days_max)
# Fnu_m2_in = eqn.Fnu_m2(p, z, epsilonB, n0, E_52, d_L28)

# plt.scatter(t_m2_days, Fnu_m2_in, marker='s', label='Fnu_m2', color='limegreen')

# def functosolve3(t):
#     return eqn.nu_sa5(p, z, epsilonE, epsilonB, n0, E_52, t) - nu
# t_sa5_days = brentq(functosolve3, t_days_min, t_days_max)
# Fnu_sa5_in = eqn.Fnu_sa5(p, z, epsilonE, epsilonB, E_52, t_sa5_days, d_L28)

# plt.scatter(t_sa5_days, Fnu_sa5_in, marker='P', label='Fnu_sa5', color='royalblue')

# obsdata = np.genfromtxt('../../multiscatter/combinedimfitsummaries.csv', unpack=False, skip_header=2, delimiter = ',',
#             dtype={'names': ('grb','id', 'date','trigger','duration','fint', 'finterr',
#              'fpk','fpkerr', 'ra', 'dec','raerr','decerr',
#              'conmaj','conmin','conpa','conmajerr','conminerr',
#              'conpaerr','deconmaj','deconmin','deconpa','deconmajerr',
#              'decondeconminerr','deconpaerr','freq','rms'),
#               'formats': ('U32','U32','U32','U32','f8','f8','f8',
#               'f8','f8','f8','f8','f8','f8',
#               'f8','f8','f8','f8','f8',
#               'f8','f8','f8','f8','f8',
#               'f8','f8','f8','f8')})
# 
# datestrings = obsdata['date'][obsdata['grb'] == "GRB200219A"]
# durations = obsdata['duration'][obsdata['grb'] == "GRB200219A"]
# triggerdate = datetime.datetime.strptime(obsdata['trigger'][obsdata['grb'] == "GRB200219A"][0], "%Y-%m-%dT%H:%M:%S.%f")
# xdate = np.array([((datetime.datetime.strptime(d, "%Y-%m-%dT%H:%M:%S.%f")
#     - triggerdate).total_seconds()+(dur/2))/3600/24 for d,dur in zip(datestrings, durations)])
# print(xdate, 3*obsdata['rms'][obsdata['grb'] == "GRB200219A"])
# plt.scatter(xdate, 1e3*3*obsdata['rms'][obsdata['grb'] == "GRB200219A"], marker='v', label='Obs UL', color='black')
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(t_days_min*0.5,t_days_max*1.5)
# ax.set_ylim(np.amin(fluxes[fluxes!=0])*0.5, np.amax(np.array([np.amax(fluxes)*1.5, Fnu_m4_in*1.5, Fnu_m2_in*1.5])))
ax.set_ylim(np.amin(fluxes[fluxes!=0])*0.5, np.amax(fluxes)*1.5)
ax.set_xlabel('t_days')
ax.set_ylabel('F ($\mu$ Jy)')
plt.title('Lightcurve at '+str(nu/1e9)+' GHz')

lines = ['solid', 'dotted', 'dashed', 'dashdot' , (0, (1, 10)), (0, (5,10))]
index = 0

for t, tname in zip(transitionT, transitionName):
    plt.axvline(x=t, label= tname, ls=lines[index])
    index = index + 1
ax.legend(fontsize=6)
plt.savefig('lightcurve.png')
plt.close()
fig= plt.figure()
plt.scatter(t_days, np.full(t_days.shape,nua1_arr), label='nua1', s=5)
plt.scatter(t_days, nua3_arr, label='nua3', s=5)
plt.scatter(t_days, num_arr, label='num', s=5)

ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(t_days_min*0.5,t_days_max*1.5)

ax.legend(fontsize=6)
plt.savefig('breakEvolution.png')
# PLOT SPECTRA AT GIVEN TIMES LOOK FOR MISSING SEGMENTS ? Maybe not?
# Why is nu_m not crossing nu_obs properly ? 
