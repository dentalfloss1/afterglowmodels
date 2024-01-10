import numpy as np 

### https://iopscience.iop.org/article/10.1086/338966/pdf
### FOR ISM ENVIRONMENT ###
## Power law segments 
def PLS_A(p, z, epsilon_B, n0, E_52, t_days, d_L28, nu_14 ):
    return 1.18 * (4.59 - p) * 1.0e8 * (1.+z)**(9./4.) * epsilon_B**(-1./4.) * n0**(-1./2.) * E_52**(1./4.) * t_days**(5./4.) * d_L28**(-2.) * nu_14**(5./2.)

def PLS_B(p, z, epsilon_e, n0, E_52, t_days, d_L28, nu_14 ):
    epsilon_bar_e = epsilon_e * (p-2.) / (p-1.)
    return 4.20 * ( (3.*p + 2.) / (3.*p - 1.) ) * 1.0e9 * (1.+z)**(5./2.) * epsilon_bar_e * n0**(-1./2.) * E_52**(1./2.) * t_days**(1./2.) * d_L28**(-2.) * nu_14**2.

def PLS_C(p, z, epsilon_B, n0, E_52, t_days, d_L28, nu_14 ):
    return 8.01e5 * (1.+z)**(27./16.) * epsilon_B**(-1./4.) * n0**(-5./16.) * E_52**(7./16.) * t_days**(11./16.) * d_L28**(-2.) * nu_14**(11./8.)

def PLS_D(p, z, epsilon_e, epsilon_B, n0, E_52, t_days, d_L28, nu_14 ):
    epsilon_bar_e = epsilon_e * (p-2.) / (p-1.)
    return 27.9 * ( (p-1.) / (3.*p-1.)) * (1.+z)**(5./6.) * epsilon_bar_e**(-2./3.) * epsilon_B**(1./3.) * n0**(1./2.) * E_52**(5./6.) * t_days**(1./2.) * d_L28**(-2.) * nu_14**(1./3.)

def PLS_E(p, z, epsilon_B, n0, E_52, t_days, d_L28, nu_14 ):
    return 73.0 * (1.+z)**(7./6.) * epsilon_B * n0**(5./6.) * E_52**(7./6.) * t_days**(1./6.) * d_L28**(-2.) * nu_14**(1./3.)

def PLS_F(p, z, epsilon_B, n0, E_52, t_days, d_L28, nu_14):
    return 6.87 * (1.+z)**(3./4.) * epsilon_B**(-1./4.) * E_52**(3./4.) * t_days**(-1./4.) * d_L28**(-2.) * nu_14**(-1./2.)

def PLS_G(p, z, epsilon_e, epsilon_B, n0, E_52, t_days, d_L28, nu_14 ):
    epsilon_bar_e = epsilon_e * (p-2.) / (p-1.)
    return 0.461 * (p-0.04) * np.exp(2.53*p) * (1.+z)**((3.+p)/4.) * epsilon_bar_e**(p-1.) * epsilon_B**((1.+p)/4.) * n0**(1./2.) * E_52**((3.+p)/4.) * t_days**(3.*(1.-p)/4.) * d_L28**(-2.) * nu_14**((1.-p)/2.)

def PLS_H(p, z, epsilon_e, epsilon_B, n0, E_52, t_days, d_L28, nu_14 ):
    epsilon_bar_e = epsilon_e * (p-2.) / (p-1.)
    return 0.855 * (p-0.98) * np.exp(1.95*p) * (1.+z)**((2.+p)/4.) * epsilon_bar_e**(p-1.) * epsilon_B**((p-2.)/4.) * E_52**((2.+p)/4.) * t_days**((2.-3.*p)/4.) * d_L28**(-2.) * nu_14**(-p/2.)

### Break Frequencies 
def nu_sa1(p, z, epsilon_e, epsilon_B, n0, E_52):
    epsilon_bar_e = epsilon_e * (p-2.) / (p-1.)
    return 1.24 * ( (p-1.)**(3./5.) / (3.*p + 2.)**(3./5.) ) * 1.0e9 * (1.+z)**(-1.) * epsilon_bar_e**(-1.) * epsilon_B**(1./5.) * n0**(3./5.) * E_52**(1./5.)

def nu_m2(p, z, epsilon_e, epsilon_B, E_52, t_days):
    epsilon_bar_e = epsilon_e * (p-2.) / (p-1.)
    return 3.73 * (p-0.67) * 1.0e15 * (1.+z)**(1./2.) * E_52**(1./2.) * epsilon_bar_e**2. * epsilon_B**(1./2.) * t_days**(-3./2.)

def nu_c3(p, z, epsilon_B, n0, E_52, t_days):
    return 6.37 * (p-0.46) * 1.0e13 * np.exp(-1.16*p) * (1.+z)**(-1./2.) * epsilon_B**(-3./2.) * n0**(-1.) * E_52**(-1./2.) * t_days**(-1./2.)

def nu_m4(p, z, epsilon_e, epsilon_B, E_52, t_days):
    epsilon_bar_e = epsilon_e * (p-2.) / (p-1.)
    return 5.04 * (p-1.22) * 1.0e16 * (1.+z)**(1./2.) * epsilon_bar_e**2. * epsilon_B**(1./2.) * E_52*(1./2.) * t_days**(-3./2.)

def nu_sa5(p, z, epsilon_e, epsilon_B, n0, E_52, t_days):
    epsilon_bar_e = epsilon_e * (p-2.) / (p-1.)
    return 3.59 * (4.03-p) * 1.0e9 * np.exp(2.34*p) * ( (epsilon_bar_e**(4.*(p-1.)) * epsilon_B**(p+2.) * n0**4. * E_52**(p+2.)) / ((1.+z)**(6.-p) * t_days**(3.*p+2.)))**(1./(2.*(p+4.)))

def nu_sa6(p, z, epsilon_e, epsilon_B, n0, E_52, t_days):
    epsilon_bar_e = epsilon_e * (p-2.) / (p-1.)
    return 3.23 * (p-1.76) * 1.0e12 * ( (epsilon_bar_e**(4.*(p-1.)) * epsilon_B**(p-1.) * n0**2. * E_52**(p+1.)) / ((1.+z)**(7.-p) * t_days**(3.*(p+1.))) )**(1./(2.*(p+5.)))

def nu_ac7(p, z, epsilon_e, epsilon_B, n0, E_52, t_days):
    epsilon_bar_e = epsilon_e * (p-2.) / (p-1.)
    return 1.12 * ((3.*p-1.)**(8./5.) / (3.*p + 2.)**(8./5.)) * 1.0e8 * (1.+z)**(-13./10.) * epsilon_bar_e**(-8./5.) * epsilon_B**(-2./5.) * n0**(3./10.) * E_52**(-1./10.) * t_days**(3./10.)
    
def nu_sa8(z, n0, E_52, t_days):
    return 1.98e11 * (1.+z)**(-1./2.) * n0**(1./6.) * E_52**(1./6.) * t_days**(-1./2.)

def nu_m9(p, z, epsilon_e, epsilon_B, E_52, t_days):
    epsilon_bar_e = epsilon_e * (p-2.) / (p-1.)
    return 3.94 * (p-0.74) * 1.e15 * (1.+z)**(1./2.) * epsilon_bar_e**2. * epsilon_B**(1./2.) * E_52**(1./2.) * t_days**(-3./2.)

def nu_sa10(z, epsilon_B, n0, E_52, t_days):
    return 1.32e10 * (1.+z)**(-1./2.) * epsilon_B**(6./5.) * n0**(11./10.) * E_52**(7./10.) * t_days**(-1./2.)

def nu_c11(z, epsilon_B, n0, E_52, t_days):
    return 5.86e12 * (1.+z)**(-1./2.) * epsilon_B**(-3./2.) * n0**(-1.) * E_52**(-1./2.) * t_days**(-1./2.)

    ### Flux at break Frequencies 
def Fnu_sa1(p, z, epsilon_e, epsilon_B, n0, E_52, t_days, d_L28):
    epsilon_bar_e = epsilon_e * (p-2.) / (p-1.)
    return 0.647 * (p-1.)**(6./5.) / ( (3.*p-1.) * (3.*p+2.)**(1./5.)) * (1.+z)**(1./2.) * epsilon_bar_e**(-1.) * epsilon_B**(2./5.) * n0**(7./10.) * E_52**(9./10.) * t_days**(1./2.) * d_L28**(-2.)

def Fnu_m2(p, z, epsilon_B, n0, E_52, d_L28):
    return 9.93 * (p+0.14) * (1.+z) * epsilon_B**(1./2.) * n0**(1./2.) * E_52 * d_L28**(-2.)

def Fnu_c3(p, z, epsilon_e, epsilon_B, n0, E_52, t_days, d_L28):
    epsilon_bar_e = epsilon_e * (p-2.) / (p-1.)
    return 4.68 * np.exp(4.82 * (p-2.5)) * 1.0e3 * (1.+z)**((p+1.)/2.) * epsilon_bar_e**(p-1.) * epsilon_B**(p-(1./2.)) * n0**(p/2.) * E_52**((p+1.)/2.) * t_days**((1.-p)/2.) * d_L28**(-2.)

def Fnu_m4(p, z, epsilon_e, epsilon_B, n0, E_52, t_days, d_L28):
    epsilon_bar_e = epsilon_e * (p-2.) / (p-1.)
    return 3.72 * (p-1.79) * 1.0e15 * (1.+z)**(7./2.) * epsilon_bar_e**5. * epsilon_B * n0**(-1./2.) * E_52**(3./2.) * t_days**(-5./2.) * d_L28**(-2.)

def Fnu_sa5(p, z, epsilon_e, epsilon_B, E_52, t_days, d_L28):
    epsilon_bar_e = epsilon_e * (p-2.) / (p-1.)
    return 20.8 * (p-1.53) * np.exp(2.56*p) * d_L28**(-2.) * ( ( (1.+z)**(7.*p+3.) * epsilon_B**(2.*p+3.) * E_52**(3.*p+7.) ) / ( epsilon_bar_e**(10.*(1.-p)) * t_days**(5.*(p-1.)) ) )**(1./(2.*(p+4.)))

def Fnu_sa6(p, z, epsilon_e, epsilon_B, n0, E_52, t_days, d_L28):
    epsilon_bar_e = epsilon_e * (p-2.) / (p-1.)
    return 76.9 * (p-1.08) * np.exp(2.06*p) * d_L28**(-2.) * ( ( (1.+z)**(7.*p+5.) * epsilon_B**(2.*p-5.) * E_52**(3.*p+5.) ) / ( epsilon_bar_e**(10.*(1.-p)) * n0**p * t_days**(5.*(p-1.)) ) )**(1./(2.*(p+5.)))

def Fnu_ac7(p, z, epsilon_e, epsilon_B, n0, E_52, t_days, d_L28):
    epsilon_bar_e = epsilon_e * (p-2.) / (p-1.)
    return 5.27 * ( (3.*p-1.)**(11./5.) / (3.*p+2.)**(11./5.) ) * 1.0e-3 * (1.+z)**(-1./10.) * epsilon_bar_e**(-11./5.) * epsilon_B**(-4./5.) * n0**(1./10.) * E_52**(3./10.) * t_days**(11./10.) * d_L28**(-2.)

def Fnu_sa8(z, epsilon_B, n0, E_52, d_L28):
    return 154. * (1.+z) * epsilon_B**(-1./4.) * n0**(-1./12.) * E_52**(2./3.) * d_L28**(-2.)

def Fnu_m9(p, z, epsilon_e, epsilon_B, E_52, t_days, d_L28):
    epsilon_bar_e = epsilon_e * (p-2.) / (p-1.)
    return 0.221 * (6.27-p) * (1.+z)**(1./2.) * epsilon_bar_e**(-1.) * epsilon_B**(-1./2.) * E_52**(1./2.) * t_days**(1./2.) * d_L28**(-2.)

def Fnu_sa10(z, epsilon_B, n0, E_52, d_L28):
    return 3.72 * (1.+z) * epsilon_B**(7./5.) * n0**(6./5.) * E_52**(7./5.) * d_L28**(-2.)

def Fnu_c11(z, epsilon_B, n0, E_52, d_L28):
    return 28.4 * (1.+z) * epsilon_B**(1./2.) * n0**(1./2.) * E_52 * d_L28**(-2.)

