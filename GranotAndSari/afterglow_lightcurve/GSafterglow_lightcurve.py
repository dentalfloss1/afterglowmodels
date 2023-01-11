import numpy as np 
import configparser
import matplotlib.pyplot as plt
import datetime 
import sys 
sys.path.append('../')
import GS_equations as gseqn
import vdH_equations as vdheqn
def vdH_fluxes(epsilonB, epsilonE, n0, E_52, d_L28, z, t_days):

    t_sec = t_days*3600*24
    D_L = d_L28*1e28
    E = E_52*1e52

    
    nuc_arr = vdheqn.nuc(epsilonB, n0, E, D_L, z, t_sec)
    num_arr = vdheqn.num(epsilonB, epsilonE, n0, E, D_L, z, t_sec, p) 
    nua1_arr = vdheqn.nua1(epsilonB, epsilonE, n0, E, D_L, z, t_sec, p)
    nua2_arr = vdheqn.nua2(epsilonB, epsilonE, n0, E, D_L, z, t_sec)
    nua3_arr = vdheqn.nua3(epsilonB, epsilonE, n0, E, D_L, z, t_sec, p)
    
    
    
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
    
    
    fluxes[spec5segmentCcond] = vdheqn.F1( epsilonB, epsilonE, n0, E, D_L, z, t_sec[spec5segmentCcond], nu)
    fluxes[spec5segmentEcond] = vdheqn.F2( epsilonB, epsilonE, n0, E, D_L, z, t_sec[spec5segmentEcond], nu)
    fluxes[spec5segmentFcond] = vdheqn.F3( epsilonB, n0, E, D_L, z, t_sec[spec5segmentFcond], nu)
    fluxes[spec5segmentHcond] = vdheqn.F4(epsilonB, epsilonE, n0, E, D_L, z, t_sec[spec5segmentHcond] ,p, nu)
    
    fluxes[spec1segmentBcond] = vdheqn.F5(epsilonB, epsilonE, n0, E, D_L, z, t_sec[spec1segmentBcond], p, nu)
    fluxes[spec1segmentDcond] = vdheqn.F6(epsilonB, epsilonE, n0, E, D_L, z, t_sec[spec1segmentDcond] , p, nu)
    fluxes[spec1segmentGcond] = vdheqn.F7(epsilonB, epsilonE, n0, E, D_L, z, t_sec[spec1segmentGcond] ,p, nu)
    fluxes[spec1segmentHcond] = vdheqn.F8(epsilonB, epsilonE, n0, E, D_L, z, t_sec[spec1segmentHcond] ,p, nu)
    
    fluxes[spec2segmentBcond] = vdheqn.F9(epsilonB, epsilonE, n0, E, D_L, z, t_sec[spec2segmentBcond] ,p, nu)
    fluxes[spec2segmentAcond] = vdheqn.F10(epsilonB, epsilonE, n0, E, D_L, z, t_sec[spec2segmentAcond] ,p, nu)
    fluxes[spec2segmentGcond] = vdheqn.F11(epsilonB, epsilonE, n0, E, D_L, z, t_sec[spec2segmentGcond] ,p, nu)
    fluxes[spec2segmentHcond] = vdheqn.F12(epsilonB, epsilonE, n0, E, D_L, z, t_sec[spec2segmentHcond] ,p, nu)
    
    fluxes = np.copy(fluxes)*1e26

    return fluxes, transitionT, transitionName

def gs_fluxes(p, z, epsilon_e, epsilon_B, n0, E_52, t_days, nu):
    """Takes parameters as input and outputs the flux along with 
    the transition times between the spectra. This goes 
    according to G&S2002."""

    nu_14 = nu/1e14
    nu_sa1_arr = gseqn.nu_sa1(p, z, epsilon_e, epsilon_B, n0, E_52)
    nu_m2_arr = gseqn.nu_m2(p, z, epsilon_e, epsilon_B, E_52, t_days)
    nu_c3_arr = gseqn.nu_c3(p, z, epsilon_B, n0, E_52, t_days)
    nu_m4_arr = gseqn.nu_m4(p, z, epsilon_e, epsilon_B, E_52, t_days)
    nu_sa5_arr = gseqn.nu_sa5(p, z, epsilon_e, epsilon_B, n0, E_52, t_days)
    nu_sa6_arr = gseqn.nu_sa6(p, z, epsilon_e, epsilon_B, n0, E_52, t_days)
    nu_ac7_arr = gseqn.nu_ac7(p, z, epsilon_e, epsilon_B, n0, E_52, t_days)
    nu_sa8_arr = gseqn.nu_sa8(z, n0, E_52, t_days)
    nu_m9_arr = gseqn.nu_m9(p, z, epsilon_e, epsilon_B, E_52, t_days)
    nu_sa10_arr = gseqn.nu_sa10(z, epsilon_B, n0, E_52, t_days)
    nu_c11_arr = gseqn.nu_c11(z, epsilon_B, n0, E_52, t_days)
    spectrum5_indices = (nu_ac7_arr < nu_sa10_arr) & (nu_sa10_arr < nu_c11_arr) & (nu_c11_arr < nu_m9_arr)
    
    
    spectrum1_indices = (nu_sa1_arr < nu_m2_arr) & (nu_m2_arr < nu_c3_arr) 
    
    spectrum2_indices = (nu_m4_arr < nu_sa5_arr) & (nu_sa5_arr < nu_c3_arr)
    # spectrum2_indices = ~np.zeros(spectrum5_indices.shape, dtype=bool)
    # spectrum1_indices= np.zeros(spectrum5_indices.shape, dtype=bool)
    spec5segmentBcond = (nu < nu_ac7_arr) & spectrum5_indices
    spec5segmentCcond = (nu > nu_ac7_arr) & (nu < nu_sa10_arr) & spectrum5_indices
    spec5segmentEcond = (nu > nu_sa10_arr) & (nu < nu_c11_arr) & spectrum5_indices
    spec5segmentFcond = (nu > nu_c11_arr) & (nu < nu_m9_arr) & spectrum5_indices
    spec5segmentHcond = (nu > nu_m9_arr) & spectrum5_indices
    
    spec1segmentBcond = (nu < np.full(t_days.shape, nu_sa1_arr)) & spectrum1_indices
    spec1segmentDcond = (nu > np.full(t_days.shape, nu_sa1_arr)) & (nu < nu_m2_arr) & spectrum1_indices
    spec1segmentGcond = (nu > nu_m2_arr) & (nu < nu_c3_arr) & spectrum1_indices
    spec1segmentHcond = (nu > nu_c3_arr) & spectrum1_indices
    
    spec2segmentBcond = (nu < nu_m4_arr) & spectrum2_indices
    # spec2segmentAcond = (nu > nu_m4_arr)  & spectrum2_indices
    spec2segmentAcond = (nu > nu_m4_arr) & (nu < nu_sa5_arr) & spectrum2_indices
    # spec2segmentGcond = (nu > nu_c3_arr) & spectrum2_indices
    spec2segmentGcond = (nu > nu_sa5_arr) & (nu < nu_c3_arr) & spectrum2_indices
    spec2segmentHcond = (nu > nu_c3_arr) & spectrum2_indices
    
    
    
    condlist = [spec5segmentBcond, spec5segmentCcond, spec5segmentEcond, spec5segmentFcond, spec5segmentHcond, spec1segmentBcond, spec1segmentDcond, spec1segmentGcond, spec1segmentHcond, spec2segmentBcond, spec2segmentAcond, spec2segmentGcond, spec2segmentHcond]
    condnamelist = ['5B', '5C', '5E', '5F', '5H', '1B', '1D', '1G', '1H' , '2B', '2A' ,'2G', '2H']
    transitionT = []
    transitionName = []
    for c in condlist:
        if np.sum(c) != 0 :
            leftregion = np.where(c==True)[0][-1]
            transitionT.append(t_days[leftregion])
    rgnattime = np.zeros(t_days.shape, dtype='U2')
    
    for i in range(len(t_days)):
        for j in range(len(condlist)):
            if condlist[j][i] == True:
                rgnattime[i] = condnamelist[j]
            else: 
                pass
    
    print('Tdays, nu, nu_sa1, nu_sa5_arr, nu_m2_arr, nu_m4_arr, rgn')
    for i in range(len(t_days)):
        print("{:e}".format(t_days[i]), "{:e}".format(nu), "{:e}".format(nu_sa1_arr), "{:e}".format(nu_sa5_arr[i]),  "{:e}".format(nu_m2_arr[i]), "{:e}".format(nu_m4_arr[i]), rgnattime[i], sep=',')
    
    
    _, idx = np.unique(rgnattime[rgnattime != ''], return_index=True)
    uniquergn = rgnattime[rgnattime != ''][np.sort(idx)]
    print(uniquergn)
    for i in range(0,len(uniquergn)-1):
        transitionName.append(uniquergn[i]+"|"+uniquergn[i+1])
    # for n,t in zip(nu_m4_arr, t_days):
    #     print(t,'{:e}'.format(nu), '{:e}'.format(n))
    
    
    
    fluxes = np.zeros(t_days.shape, dtype=float)
    
    
    fluxes[spec5segmentBcond] = gseqn.PLS_B(p, z, epsilon_e, n0, E_52, t_days[spec5segmentBcond], d_L28, nu_14 )
    fluxes[spec5segmentCcond] = gseqn.PLS_C(p, z, epsilon_B, n0, E_52, t_days[spec5segmentCcond], d_L28, nu_14 )
    fluxes[spec5segmentEcond] = gseqn.PLS_E(p, z, epsilon_B, n0, E_52, t_days[spec5segmentEcond], d_L28, nu_14 )
    fluxes[spec5segmentFcond] = gseqn.PLS_F(p, z, epsilon_B, n0, E_52, t_days[spec5segmentFcond], d_L28, nu_14 )
    fluxes[spec5segmentHcond] = gseqn.PLS_H(p, z, epsilon_e, epsilon_B, n0, E_52, t_days[spec5segmentHcond], d_L28, nu_14 )
    
    fluxes[spec1segmentBcond] = gseqn.PLS_B(p, z, epsilon_e, n0, E_52, t_days[spec1segmentBcond], d_L28, nu_14 )
    fluxes[spec1segmentDcond] = gseqn.PLS_D(p, z, epsilon_e, epsilon_B, n0, E_52, t_days[spec1segmentDcond], d_L28, nu_14 )
    fluxes[spec1segmentGcond] = gseqn.PLS_G(p, z, epsilon_e, epsilon_B, n0, E_52, t_days[spec1segmentGcond], d_L28, nu_14 )
    fluxes[spec1segmentHcond] = gseqn.PLS_H(p, z, epsilon_e, epsilon_B, n0, E_52, t_days[spec1segmentHcond], d_L28, nu_14 )
    
    fluxes[spec2segmentBcond] = gseqn.PLS_B(p, z, epsilon_e, n0, E_52, t_days[spec2segmentBcond], d_L28, nu_14 )
    fluxes[spec2segmentAcond] = gseqn.PLS_A(p, z, epsilon_B, n0, E_52, t_days[spec2segmentAcond], d_L28, nu_14 )
    fluxes[spec2segmentGcond] = gseqn.PLS_G(p, z, epsilon_e, epsilon_B, n0, E_52, t_days[spec2segmentGcond], d_L28, nu_14 )
    fluxes[spec2segmentHcond] = gseqn.PLS_H(p, z, epsilon_e, epsilon_B, n0, E_52, t_days[spec2segmentHcond], d_L28, nu_14 )
    return fluxes, transitionT, transitionName

def scatterplot(t_days, fluxes, transitionT, transitionName, fig, label, color):
    
    plt.scatter(t_days, fluxes, marker='o',  label=label, color=color, s=5)
    
    
    lines = ['solid', 'dotted', 'dashed', 'dashdot' , (0, (1, 10)), (0, (5,10))]
    index = 0
    print(transitionT)
    print(transitionName)
    
    for t, tname in zip(transitionT, transitionName):
        plt.axvline(x=t, label= tname, ls=lines[index], color=color)
        index = index + 1
if __name__ == "__main__":
    # READ in parameters from parameters.config
    # Follows Granot & Sari 2002
    # https://ui.adsabs.harvard.edu/abs/2002ApJ...568..820G/abstract

    params = configparser.ConfigParser()
    params.read('parameters.config')
    
    # GRB PARAMS
    p = float(params['PARAMETERS']['p'])
    z = float(params['PARAMETERS']['z'])
    epsilon_B = float(params['PARAMETERS']['epsilon_B'])
    epsilon_e = float(params['PARAMETERS']['epsilon_e'])
    n0 = float(params['PARAMETERS']['n0'])
    E_52 = float(params['PARAMETERS']['E_52'])
    t_days_min = float(params['PARAMETERS']['t_days_min'])
    t_days_max = float(params['PARAMETERS']['t_days_max'])
    d_L28 = float(params['PARAMETERS']['d_L28'])
    nu = float(params['PARAMETERS']['nu'])
    
    # Computation params
    res = int(float(params['PARAMETERS']['res']))
    t_days = np.geomspace(t_days_min, t_days_max, num=res)
    
    #gs_fluxes does the heavy lifting of doing the conditionals and determining where to put the breaks
    gsfluxes, gstransitionT, gstransitionName = gs_fluxes(p, z, epsilon_e, epsilon_B, n0, E_52, t_days, nu)
    vdhfluxes, vdhtransitionT, vdhtransitionName = vdH_fluxes(epsilon_B, epsilon_e, n0, E_52, d_L28, z, t_days)

    boxfitdata = np.loadtxt('boxfitlightcurve.txt', delimiter=',',dtype={'names': ('i', 't', 'nu', 'F'), 'formats': (int, float, float, float)})

    fig = plt.figure()
    ax = scatterplot(t_days, gsfluxes, gstransitionT, gstransitionName, fig, 'G&S', 'navy')
    ax = scatterplot(t_days, vdhfluxes, vdhtransitionT, vdhtransitionName, fig, 'vdH', 'seagreen')
    minval = np.minimum(np.amin(gsfluxes[gsfluxes!=0]), np.amin(vdhfluxes[vdhfluxes!=0]))
    maxval = np.maximum(np.amax(gsfluxes)*1.5, np.amax(vdhfluxes)*1.5)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(t_days_min*0.5,t_days_max*1.5)
    ax.set_ylim(minval, maxval)
    ax.set_xlabel('t_days')
    ax.set_ylabel('F (mJy)')
    plt.title('Lightcurve at '+str(nu/1e9)+' GHz')
    plt.scatter(boxfitdata['t']/3600/24, boxfitdata['F'], marker='*',  label='boxfit', color='darkorange', s=5)
    obsdata = np.genfromtxt('../../multiscatter/combinedimfitsummaries.csv', unpack=False, skip_header=2, delimiter = ',',
                dtype={'names': ('grb','id', 'date','trigger','duration','fint', 'finterr',
                 'fpk','fpkerr', 'ra', 'dec','raerr','decerr',
                 'conmaj','conmin','conpa','conmajerr','conminerr',
                 'conpaerr','deconmaj','deconmin','deconpa','deconmajerr',
                 'decondeconminerr','deconpaerr','freq','rms'),
                  'formats': ('U32','U32','U32','U32','f8','f8','f8',
                  'f8','f8','f8','f8','f8','f8',
                  'f8','f8','f8','f8','f8',
                  'f8','f8','f8','f8','f8',
                  'f8','f8','f8','f8')})
    
    datestrings = obsdata['date'][obsdata['grb'] == "GRB200219A"]
    durations = obsdata['duration'][obsdata['grb'] == "GRB200219A"]
    triggerdate = datetime.datetime.strptime(obsdata['trigger'][obsdata['grb'] == "GRB200219A"][0], "%Y-%m-%dT%H:%M:%S.%f")
    xdate = np.array([((datetime.datetime.strptime(d, "%Y-%m-%dT%H:%M:%S.%f")
        - triggerdate).total_seconds()+(dur/2))/3600/24 for d,dur in zip(datestrings, durations)])
    # print(xdate, 3*obsdata['rms'][obsdata['grb'] == "GRB200219A"])
    plt.scatter(xdate, 1e3*3*obsdata['rms'][obsdata['grb'] == "GRB200219A"], marker='v', label='Obs UL', color='black')
    ax.legend(fontsize=6)
    plt.savefig('lightcurve.png')
    plt.close()
