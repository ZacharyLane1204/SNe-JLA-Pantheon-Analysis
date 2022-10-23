
# %%
import sys
import numpy as np
from scipy import integrate,optimize
import matplotlib.pyplot as plt

# Updated 07/10/2022 to fix a bug in the code that was affecting the distance moduli of a single data point

class flrw:
    
    def __init__(self, om0, oml, H0=66.7):
        self.om0 = om0
        self.oml = oml
        self.c = 2.99792458e5
        self.dH = self.c/H0

    @property
    def omk(self):
        return 1.-self.om0-self.oml

    def _integrand(self, zcmb):
        """
        Computes H0/H(z) where zcmb is redshift measured in cmb frame

        """
        z1 = 1.+zcmb
        return 1/np.sqrt(self.om0*z1**3 + self.omk*z1**2 + self.oml)

    def dL(self, zcmb):
        """
        Computes dL/dH, dH=c/H0 measured in cmb frame

        """
        Ok = self.omk
        Om = self.om0
        Ol = self.oml

        z = zcmb
        z1 = 1.+z
        if Ok < 0: # closed
            I,err = integrate.quad(self._integrand, 0.0, z)
            q = np.sqrt(np.absolute(Ok))
            return z1*np.sin(q*I)/q
        elif Ok > 0: # open
            if Ok == 1: # Milne
                return 0.5*(z1**2-1.) # z1*np.sinh(np.log(z1))
            else:
                I,err = integrate.quad(self._integrand, 0.0, z)
                q = np.sqrt(Ok)
                return z1*np.sinh(q*I)/q
        else: # flat
            if Om == 1: # Einstein-de Sitter 
                return 2.*z1*(1.-1./np.sqrt(z1))
            elif Ol == 1: # de Sitter
                return z1*z
            else:
                I,err = integrate.quad(self._integrand, 0.0, z)
                return z1*I

    def mu(self, zcmb):
        return 5*np.log10(self.dL(zcmb)) + 25


class timescape:
    
    def __init__(self, om0, H0=66.7):
        
        self.om0 = om0
        self.fv0 = 0.5*(np.sqrt(9-8.*self.om0)-1)                

        self.c = 2.99792458e5
        self.dH = self.c/H0
        self.t0 = (2.+self.fv0)/3 # age of universe
        
        self._y0 = self.t0**(1./3)
        self._hf = (4.*self.fv0**2+self.fv0+4.)/2/(2.+self.fv0) # H0/barH0
        self._b = 2*(1.-self.fv0)*(2.+self.fv0)/9/self.fv0 # b*barH0

    def _z1(self, t):
        fv = self.fv_t(t)
        return (2.+fv)*fv**(1./3)/3/t/self.fv0**(1./3)

    def tex(self, zcmb):
        """
        Get time t explicitly by inverting func(t,z)=0 for zcmb
        where zcmb is redshift measured in cmb frame
        Note t units: 1/H0

        """
        def f(t,z): return self._z1(t)-(1.+z)
        
        # Root must be enclosed in [a,b]
        a = 0.01
        b = 1.1 #t=0.93/Hbar0 (for fv0=0.778)
        try:
            root = optimize.brentq(f, a, b, args=(zcmb,), maxiter=400)
        except ValueError:
            sys.exit('z = {0:1.3f}\nf(a) = {1:1.3f}\nf(b) = {2:1.3f}'.format(zcmb, f(a,zcmb), f(b,zcmb)))
        return root

    def _yint(self, Y):
        """
        Compute \mathcal{F}(Y), Y=t^{1/3}

        """
        bb = self._b**(1./3)
        return 2.*Y + (bb/6.)*np.log((Y+bb)**2/(Y**2-bb*Y+bb**2)) \
            + bb/np.sqrt(3)*np.arctan((2.*Y-bb)/(np.sqrt(3)*bb))

    def fv_t(self, t):
        """
        Tracker soln as fn of time

        """
        return 3.*self.fv0*t/(3.*self.fv0*t+(1.-self.fv0)*(2.+self.fv0))

    def fv_z(self, zcmb):
        """
        Tracker soln as fn of redshift

        """
        t = self.tex(zcmb)
        return 3.*self.fv0*t/(3.*self.fv0*t+(1.-self.fv0)*(2.+self.fv0))

    def dA(self, zcmb):
        """
        Angular diameter distance divided by dH=c/H0

        """
        ya = self.tex(zcmb)**(1./3) #t^{1/3}
        return ya**2 * (self._yint(self._y0)-self._yint(ya))

    def H0D(self, zcmb): #H0D/dH
        return self._hf*(1.+zcmb)*self.dA(zcmb)

    def dL(self, zcmb):
        """
        Luminosity distance, units Mpc

        """
        return self.dH*(1.+zcmb)*self.H0D(zcmb)
    
    def mu(self, zcmb):
        """
        Distance modulus

        """
        return 5*np.log10(self.dL(zcmb)) + 25

    
if __name__ == '__main__':

    
    def Mu_output(Om0, Ol, fv0):
    
        
        Om = 0.5*(1-fv0)*(2+fv0)
        milne = 0
        # compute lcdm dL/dH
    
        fl = flrw(Om0,Ol)
        ml = flrw(milne,milne)
        ts = timescape(Om,H0=61.7)
        
        dist_lcdm_list = []
        dist_milne_list = []
        dist_ts_list = []
        
        z_range = np.linspace(0,1.5,401)
        
        for z in z_range:
            dist_lcdm = fl.dL(z)
            dist_lcdm_list.append(dist_lcdm)
            dist_milne = ml.dL(z)
            dist_milne_list.append(dist_milne)
            dist_ts = ts.dL(z)
            dist_ts_list.append(dist_ts)
    
    
        c = 299792.458
        H0  = 66.7
        dist_ratio = c/H0
    
        dist_lcdm_list = dist_ratio*np.array(dist_lcdm_list)
        dist_milne_list = dist_ratio*np.array(dist_milne_list)
        dist_ts_list = (61.7/66.7)*np.array(dist_ts_list)
        dist_ts_list[0] = 0
        
        mu_lcdm = 5*np.log10(dist_lcdm_list) + 25
        mu_empty = 5*np.log10(dist_milne_list) + 25
        mu_ts = 5*np.log10(dist_ts_list) + 25
        
        mu_lc_r = mu_lcdm - mu_empty
        mu_ts_r = mu_ts - mu_empty
        return mu_lc_r, mu_ts_r
    
    
    
    Om0_jla = 0.369875
    Ol_jla = 1- Om0_jla
    fv0_jla = 0.774545
    
    Om0_panthplus = 0.351118
    Ol_panthplus = 1- Om0_panthplus
    fv0_panthplus = 0.798965  

    Om0_panth = 0.387
    Ol_panth = 1- Om0_panth
    fv0_panth = 0.739
    
    Om0_panthplusPlus = 0.344500
    Ol_panthplusPlus = 1- Om0_panthplus
    fv0_panthplusPlus = 0.78947
    
    mu_lc_r_jla, mu_ts_r_jla = Mu_output(Om0_jla, Ol_jla, fv0_jla)
    mu_lc_r_panthplus, mu_ts_r_panthplus = Mu_output(Om0_panthplus, Ol_panthplus, fv0_panthplus)
    mu_lc_r_panth, mu_ts_r_panth = Mu_output(Om0_panth, Ol_panth, fv0_panth)
    mu_lc_r_panthplusPlus, mu_ts_r_panthplusPlus = Mu_output(Om0_panthplusPlus, Ol_panthplusPlus, fv0_panthplusPlus)
    
    mu_lc_r_jla[0] = 0
    mu_ts_r_jla[0] = 0
    mu_lc_r_panthplus[0] = 0
    mu_ts_r_panthplus[0] = 0

    mu_lc_r_panth[0] = 0
    mu_ts_r_panth[0] = 0

    z_range = np.linspace(0,1.5,401)
    
    plt.rcParams['font.size']= 16
    plt.figure(figsize = (9,5.5))
    plt.axhline(0, linestyle = 'dotted', alpha = 0.7, color = 'k', linewidth = 1.5)
    plt.plot(z_range, mu_lc_r_jla, color = 'C1', linestyle = '--')#, label = 'Spatially Flat $\Lambda$CDM- JLA')
    plt.plot(z_range, mu_ts_r_jla, color = 'k', linestyle = '--')#, label = 'Timescape- JLA')
    # plt.plot(z_range, mu_lc_r_panthplus, color = 'C1', linestyle = '-', label = 'Spatially Flat $\Lambda$CDM- P+')
    # plt.plot(z_range, mu_ts_r_panthplus, color = 'k', linestyle = '-', label = 'Timescape- P+')
    plt.plot(z_range, mu_lc_r_panthplusPlus, color = 'C1', linestyle = '-', label = 'Spatially Flat $\Lambda$CDM')
    plt.plot(z_range, mu_ts_r_panthplusPlus, color = 'k', linestyle = '-', label = 'Timescape')
    # plt.plot(z_range, mu_lc_r_panth, color = 'purple', linestyle = 'dotted', label = 'Spatially Flat $\Lambda$CDM- P+')
    # plt.plot(z_range, mu_ts_r_panth, color = 'purple', linestyle = 'dotted', label = 'Timescape- P+')
    plt.xlim(0,1.5)
    plt.ylim(-0.1,0.1)
    plt.xlabel('Redshift $z$')
    plt.ylabel(r'$\mu - \mu_{Empty}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('mu_empty_compare.pdf', format = 'pdf', dpi = 1200)
    plt.show()


# %%
