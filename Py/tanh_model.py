import numpy as np
from pyfe.option_model import OptionModelABC
from pyfe.bsm import BsmModel


class TanhModelABC(OptionModelABC):
    vov, h, rho = 0.0, 1.0, 0.0

    def __init__(self, sigma, vov=0.0, rho=0.0, h=1.0, intr=0.0, divr=0.0, is_fwd=False):
        super().__init__(sigma, intr=intr, divr=divr, is_fwd=is_fwd)
        self.vov = vov
        self.rho = rho
        self.h = h

    def params_kw(self):
        params1 = super().params_kw()
        params2 = {"vov": self.vov, "h": self.h, "rho": self.rho}
        return {**params1, **params2}

    @staticmethod
    def xx_z(zz, rho):  # return x(z)/z
        xx_zz = np.zeros(zz.size)
        rho2 = rho*rho
        yy = np.sqrt(1 + zz*(zz - 2*rho))

        ind = (abs(zz) < 1e-5)
        xx_zz[ind] = 1 + zz[ind]*((rho/2) + zz[ind]*((1/2*rho2 - 1/6) + 1/8*(5*rho2 - 3)*rho*zz[ind]))
        ind = (zz >= 1e-5)
        xx_zz[ind] = np.log((yy[ind] + (zz[ind] - rho))/(1 - rho))/zz[ind]
        ind = (zz <= -1e-5)
        xx_zz[ind] = np.log((1 + rho)/(yy[ind] - (zz[ind] - rho)))/zz[ind]

        return xx_zz


class TanhModelMC(TanhModelABC):
    """
    Monte Carlo simulation of the tanh model
    """

    def set_mc_params(self, dt, n_paths=10000, method="Milstein", antithetic=True):
        """
        Specify the parameters for Monte Carlo simulation
        :param dt: time interval
        :param n_paths: number of simulation trials
        :param method: "Log" or "Euler" or "Milstein"
        :param antithetic: True if use antithetic else False
        """
        self.dt = dt
        self.n_paths = n_paths
        self.method = method
        self.antithetic = antithetic

    def price(self, strike, spot, texp, cp_sign=1):
        """
        Give the price of European option under the tanh model using Monte Carlo simulation
        :param strike: strike
        :param spot: spot or forward consistent with self.is_fwd
        :param texp: time to expiration
        :param cp_sign: 1 for call and -1 for put
        :return: option price
        """
        # vol_paths
        n_steps = int(texp/self.dt)
        vov_sqrt_dt = self.vov * np.sqrt(self.dt)

        if self.antithetic:
            zz = np.random.normal(size=(int(self.n_paths/2), n_steps))
            zz = np.concatenate((zz, -zz), axis=0)
            xx = np.random.normal(size=(int(self.n_paths/2), n_steps))
            xx = np.concatenate((xx, -xx), axis=0)
        else:
            zz = np.random.normal(size=(self.n_paths, n_steps))
            xx = np.random.normal(size=(self.n_paths, n_steps))

        sigma_paths = np.ones((self.n_paths, n_steps+1))
        log_sigma_paths = np.cumsum(vov_sqrt_dt*(zz-0.5*vov_sqrt_dt), axis=1)
        sigma_paths[:, 1:] = np.exp(log_sigma_paths)
        sigma_paths = self.sigma*sigma_paths
        sigma_sqrt_dt = sigma_paths*np.sqrt(self.dt)

        # fwd_paths
        if self.is_fwd:
            fwd = spot
        else:
            fwd = spot*np.exp(texp*(self.intr-self.divr))

        ww = self.rho*zz+np.sqrt(1-self.rho**2)*xx

        if self.method == "Log":
            log_sinh_fwd = np.ones((self.n_paths, n_steps+1)) * \
                           (np.log(np.sinh(fwd/self.h)) if fwd/self.h < 710 else fwd/self.h-np.log(2))

            for i in range(1, n_steps+1):
                log_sinh_fwd[:, i] = log_sinh_fwd[:, i-1] + sigma_sqrt_dt[:, i-1] * \
                                     (ww[:, i-1]-sigma_sqrt_dt[:, i-1]/2/(1+np.exp(2*log_sinh_fwd[:, i-1])))

            log_sinh_fwd_final = log_sinh_fwd[:, -1]
            fwd_final = np.zeros(self.n_paths)

            ind = (np.absolute(log_sinh_fwd_final) < 354)
            fwd_final[~ind] = self.h*(log_sinh_fwd_final[~ind]+np.log(2))
            fwd_final[ind] = self.h*np.log(np.exp(log_sinh_fwd_final[ind])+np.sqrt(1+np.exp(2*log_sinh_fwd_final[ind])))

            price = np.mean(np.fmax(cp_sign*(fwd_final[:, None]-strike), 0), axis=0)/np.exp(self.intr*texp)

        else:
            fwd_h = np.ones((self.n_paths, n_steps+1))*fwd/self.h
            if self.method == "Euler":
                for i in range(1, n_steps+1):
                    fwd_h[:, i] = fwd_h[:, i-1] + sigma_sqrt_dt[:, i-1]*np.tanh(fwd_h[:, i-1])*ww[:, i-1]
            else:
                for i in range(1, n_steps+1):
                    fwd_h[:, i] = fwd_h[:, i-1] + sigma_sqrt_dt[:, i-1]*np.tanh(fwd_h[:, i-1]) * \
                                  (ww[:, i-1]+sigma_sqrt_dt[:, i-1]*(1-np.tanh(fwd_h[:, i-1])**2)*(ww[:, i-1]**2-1)/2)

            fwd_final = fwd_h[:, -1]*self.h
            price = np.mean(np.fmax(cp_sign*(fwd_final[:, None]-strike), 0), axis=0)/np.exp(self.intr*texp)

        return price[0] if price.size == 1 else price

    def bsm_vol(self, strike, spot, texp, cp_sign=1):
        """
        Give the implied BSM volatility under the tanh model
        :param strike: strike
        :param spot: spot or forward consistent with self.is_fwd
        :param texp: time to expiration
        :param cp_sign: 1 for call and -1 for put
        :return: implied BSM volatility
        """
        bsm_model = BsmModel(None, intr=self.intr, divr=self.divr, is_fwd=self.is_fwd)

        price = self.price(strike, spot, texp, cp_sign=cp_sign)
        vol = bsm_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)

        return vol


class TanhModelApproxVol(TanhModelABC):
    """
    Analytic approximation for the tanh model
    """

    def bsm_vol_hagan(self, strike, spot, texp):
        """
        Equivalent BSM-volatility formula for the tanh model (Hagan et al, 2002)
        :param strike: strike
        :param spot: spot or forward consistent with self.is_fwd
        :param texp: time to expiration
        :return: Analytic approximated BSM-volatility of the tanh model
        """
        fwd = spot * (1.0 if self.is_fwd else np.exp(texp*(self.intr-self.divr)))

        if isinstance(strike, (int, float)):
            strike = np.array([strike])

        fwd_strk = np.sqrt(fwd*strike)/self.h
        tanh_fwd_strk = np.tanh(fwd_strk)
        tanh_fwd_strk2 = tanh_fwd_strk**2

        pre1 = self.int_inv_locvol(fwd, strike, self.h)

        pre2alp0 = (2-3*self.rho**2)*self.vov**2/24
        pre2alp1 = (1-tanh_fwd_strk2)*self.rho*self.vov/4
        pre2alp2 = (tanh_fwd_strk2*(3*tanh_fwd_strk2+1/fwd_strk**2-2)-1)/24
        pre2 = 1.0+texp*(pre2alp0 + self.sigma*(pre2alp1 + pre2alp2*self.sigma))

        zz = self.vov/self.sigma*(fwd-strike)/self.h/tanh_fwd_strk
        if isinstance(zz, float):
            zz = np.array([zz])
        xx_zz = TanhModelABC.xx_z(zz, self.rho)

        bsm_vol = self.sigma/(pre1*xx_zz)*pre2
        return bsm_vol[0] if bsm_vol.size == 1 else bsm_vol

    @staticmethod
    def int_inv_locvol(fwd, strike, h):
        # (int from K to f 1/(h*tanh(x/h))dx) / log(f/K) = (log(sinh(f/h)) - log(sinh(K/h))) / log(f/K)
        val = np.zeros_like(strike, dtype=float)

        diff1 = (fwd-strike)/fwd
        diff2 = (fwd-strike)/h
        coth_fwd = 1/np.tanh(fwd/h)

        ind = (np.abs(fwd-strike) < 1e-6)
        val[ind] = fwd/h * (coth_fwd + (coth_fwd**2-1)/2*diff2[ind] + coth_fwd*(coth_fwd**2-1)/3*diff2[ind]**2 +
                            (3*coth_fwd**2-1)*(coth_fwd**2-1)/12*diff2[ind]**3) / \
                   (1 + diff1[ind]/2 + diff1[ind]**2/3 + diff1[ind]**3/4)

        log_sinh_fwd = np.log(np.sinh(fwd/h)) if fwd/h < 710 else fwd/h-np.log(2)

        log_sinh_strk = np.zeros_like(strike, dtype=float)
        ind1 = ~ind*(strike/h < 710)
        ind2 = ~ind*(strike/h >= 710)
        log_sinh_strk[ind1] = np.log(np.sinh(strike[ind1]/h))
        log_sinh_strk[ind2] = strike[ind2]/h-np.log(2)
        val[~ind] = (log_sinh_fwd-log_sinh_strk[~ind])/np.log(fwd/strike[~ind])

        return val

    def price(self, strike, spot, texp, cp_sign=1):
        """
        Derive option price under the tanh model from the equivalent BSM-volatility using BS formula
        :param strike: strike
        :param spot: spot or forward consistent with self.is_fwd
        :param texp: time to expiration
        :param cp_sign: 1 for call and -1 for put
        :return: option price
        """

        sigma = self.bsm_vol_hagan(strike, spot, texp)
        bsm_model = BsmModel(sigma, intr=self.intr, divr=self.divr, is_fwd=self.is_fwd)
        price = bsm_model.price(strike, spot, texp, cp_sign=cp_sign)

        return price