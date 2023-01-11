import numpy as np 
from scipy.special import gamma
alphaAD = 16./17.
betaAD = 4
X = 1
qe = 4.80320427e-10
m_e = 9.10938370e-28
m_p = 1.6726219e-24
c = 2.99792458e10


def F_nuMax(epsilonB, n, E, D_L, z):
    global alphaAD
    global X
    global qe
    global m_e
    global m_p
    global c
    return (4./3.) * np.sqrt(2) * alphaAD**(-1.) * np.pi**(-1./2.) * ((1.+X)/2.) * qe**3. * m_e**(-1.) * m_p**(-1./2.) * c**(-3.) * epsilonB**(1./2.) * n**(1./2.) * E * D_L**(-2.) * (1.+z)

def nuc(epsilonB, n, E, D_L, z, t):
    global alphaAD
    global X
    global qe
    global m_e
    global m_p
    global c
    return (81./8192.) * np.sqrt(2) * alphaAD**(1./2.) * betaAD**(3./2.) * np.pi**(-2.) * qe**(-7.) * m_e**5. * m_p**(-1) * c**(17./2.) *  epsilonB**(-3./2.) * n**(-1.) * E**(-1./2.) * t**(-1./2.) * (1+z)**(-1./2.)

def num(epsilonB, epsilonE, n, E, D_L, z, t, p):
    global alphaAD
    global X
    global qe
    global m_e
    global m_p
    global c
    return 2 * np.sqrt(2) * alphaAD**(-1./2.) * betaAD**(-3./2.) * np.pi**(-1.) * ((1+X)/2)**(-2.) * (p-2)**2. * (p-1)**(-2.) * qe * m_e**(-3.) * m_p**2. * c**(-5./2.) * epsilonE**2 * epsilonB**(1./2.) * E**(1./2.) * t**(-3./2.) * (1+z)**(1./2.)

def nua1(epsilonB, epsilonE, n, E, D_L, z, t, p):
    global alphaAD
    global X
    global qe
    global m_e
    global m_p
    global c
    # fudgeFactor =  30**(-3/5)
    return 2**(-4./5) * 3**(4./5.) * alphaAD**(-1./5.) * np.pi**(-2./5.) * ((1+X)/2)**(8./5.) * (p-2)**(-1) * (p-1)**(8/5) * (p+(2./3.))**(-3./5.) * (p+2)**(3./5.) * qe**(8./5.) * m_p**(-1.) * c**(-1.) * epsilonE**(-1.) * epsilonB**(1./5.) * n**(3./5.) * E**(1./5.) * (1+z)**(-1.)

def nua2(epsilonB, epsilonE, n, E, D_L, z, t):
    global alphaAD
    global X
    global qe
    global m_e
    global m_p
    global c
    return 2**(28./5.) * 3**(-3./5.) * alphaAD**(-7./10.) * betaAD**(-3./2.) * np.pi**(1./10.) * ((1+X)/2)**(3./5.) * qe**(28./5.) * m_e**(-4) * m_p**(1./2.) * c**(-13/2) * epsilonB**(6./5.) * n**(11/10) * E**(7/10) * t**(-1./2.) * (1+z)**(-1./2.)

def nua3(epsilonB, epsilonE, n, E, D_L, z, t, p):
    global alphaAD
    global X
    global qe
    global m_e
    global m_p
    global c
    return 2**((9*p-22)/(6*(p+4))) * 3**(8/(3*(p+4))) * alphaAD**(-((p+2)/(2*(p+4)))) * betaAD**(-((3*p+2)/(2*(p+4)))) * np.pi**(-((p+2)/(p+4))) * ((1+X)/2)**(-((2*(p+2))/(p+4))) * (gamma((p/2) + (1/3)))**(2/(p+4)) * (p-2)**((2*(p-1))/(p+4)) * (p-1)**(-((2*(p-2))/(p+4))) * (p+2)**(2/(p+4)) * qe**((p+6)/(p+4)) * m_e**(-((3*p+2)/(p+4))) * m_p**((2*(p-1))/(p+4)) * c**(-((5*p+10)/(2*(p+4)))) * epsilonE**((2*(p-1))/(p+4)) * epsilonB**((p+2)/(2*(p+4))) * n**(2/(p+4)) * E**((p+2)/(2*(p+4))) * t**(-((3*p+2)/(2*(p+4)))) * (1+z)**((p-6)/(2*(p+4)))

def F1( epsilonB, epsilonE, n, E, D_L, z, t, nu):
    fnu = F_nuMax(epsilonB, n, E, D_L, z)
    nua = nua2(epsilonB, epsilonE, n, E, D_L, z, t)
    nuc_val = nuc(epsilonB, n, E, D_L, z, t)
    return fnu * (nu/nua)**2 * (nua/nuc_val)**(1./3.)

def F2( epsilonB, epsilonE, n, E, D_L, z, t, nu):
    fnu = F_nuMax(epsilonB, n, E, D_L, z)
    nua = nua2(epsilonB, epsilonE, n, E, D_L, z, t)
    nuc_val = nuc(epsilonB, n, E, D_L, z, t)
    return fnu * (nu/nuc_val)**(1./3.)

def F3( epsilonB, n, E, D_L, z, t, nu):
    fnu = F_nuMax(epsilonB, n, E, D_L, z)
    nuc_val = nuc(epsilonB, n, E, D_L, z, t)
    return fnu * (nu/nuc_val)**(-1./2.)

def F4(epsilonB, epsilonE, n, E, D_L, z, t ,p, nu):
    fnu = F_nuMax(epsilonB, n, E, D_L, z)
    num_val = num(epsilonB, epsilonE, n, E, D_L, z, t, p)
    nuc_val = nuc(epsilonB, n, E, D_L, z, t)
    return fnu * (num_val/nuc_val)**(-1/2) * (nu/num_val)**(-p/2.)

def F5(epsilonB, epsilonE, n, E, D_L, z, t,p, nu):
    fnu = F_nuMax(epsilonB, n, E, D_L, z)
    num_val = num(epsilonB, epsilonE, n, E, D_L, z, t, p)
    nua = nua1(epsilonB, epsilonE, n, E, D_L, z, t, p)
    return fnu * (nu/nua)**2 * (nua/num_val)**(1/3.)

def F6(epsilonB, epsilonE, n, E, D_L, z, t, p, nu):
    fnu = F_nuMax(epsilonB, n, E, D_L, z)
    num_val = num(epsilonB, epsilonE, n, E, D_L, z, t, p)
    return fnu * (nu/num_val)**(1/3)

def F7(epsilonB, epsilonE, n, E, D_L, z, t ,p, nu):
    fnu = F_nuMax(epsilonB, n, E, D_L, z)
    num_val = num(epsilonB, epsilonE, n, E, D_L, z, t, p)
    return fnu * (nu/num_val)**(-(p-1)/2)

def F8(epsilonB, epsilonE, n, E, D_L, z, t ,p, nu):
    fnu = F_nuMax(epsilonB, n, E, D_L, z)
    num_val = num(epsilonB, epsilonE, n, E, D_L, z, t, p)
    nuc_val = nuc(epsilonB, n, E, D_L, z, t)
    return fnu * (nuc_val/num_val)**(-(p-1)/2) * (nu/nuc_val)**(-p/2.)

def F9(epsilonB, epsilonE, n, E, D_L, z, t ,p, nu):
    fnu = F_nuMax(epsilonB, n, E, D_L, z)
    nua = nua3(epsilonB, epsilonE, n, E, D_L, z, t, p)
    num_val = num(epsilonB, epsilonE, n, E, D_L, z, t, p)
    return fnu * (nu/num_val)**2 * (num_val/nua)**((p+4)/2)

def F10(epsilonB, epsilonE, n, E, D_L, z, t ,p, nu):
    fnu = F_nuMax(epsilonB, n, E, D_L, z)
    nua = nua3(epsilonB, epsilonE, n, E, D_L, z, t, p)
    num_val = num(epsilonB, epsilonE, n, E, D_L, z, t, p)
    return fnu * (nu/nua)**(5/2) * (num_val/nua)**((p-1)/2)

def F11(epsilonB, epsilonE, n, E, D_L, z, t ,p, nu):
    fnu = F_nuMax(epsilonB, n, E, D_L, z)
    num_val = num(epsilonB, epsilonE, n, E, D_L, z, t, p)
    nua = nua3(epsilonB, epsilonE, n, E, D_L, z, t, p)
    return fnu * (nu/num_val)**(-(p-1)/2)

def F12(epsilonB, epsilonE, n, E, D_L, z, t ,p, nu ):
    fnu = F_nuMax(epsilonB, n, E, D_L, z)
    nua = nua3(epsilonB, epsilonE, n, E, D_L, z, t, p)
    nuc_val = nuc(epsilonB, n, E, D_L, z, t)
    num_val = num(epsilonB, epsilonE, n, E, D_L, z, t, p)
    return fnu * (nu/nuc_val)**(-p/2) * (nuc_val/num_val)**(-(p-1)/2)

