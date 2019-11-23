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
        n_steps = int(texp / self.dt)
        vov_sqrt_dt = self.vov * np.sqrt(self.dt)

        if self.antithetic:
            zz = np.random.normal(size=(int(self.n_paths / 2), n_steps))
            zz = np.concatenate((zz, -zz), axis=0)
            xx = np.random.normal(size=(int(self.n_paths / 2), n_steps))
            xx = np.concatenate((xx, -xx), axis=0)
        else:
            zz = np.random.normal(size=(self.n_paths, n_steps))
            xx = np.random.normal(size=(self.n_paths, n_steps))

        sigma_paths = np.ones((self.n_paths, n_steps + 1))
        log_sigma_paths = np.cumsum(vov_sqrt_dt * (zz - 0.5 * vov_sqrt_dt), axis=1)
        sigma_paths[:, 1:] = np.exp(log_sigma_paths)
        sigma_paths = self.sigma * sigma_paths
        sigma_sqrt_dt = sigma_paths * np.sqrt(self.dt)

        # forward_paths
        if self.is_fwd:
            forward = spot
        else:
            forward = spot * np.exp(texp * (self.intr - self.divr))

        ww = self.rho * zz + np.sqrt(1 - self.rho ** 2) * xx

        if self.method == "Log":
            log_sinh_forward = np.ones((self.n_paths, n_steps + 1)) * np.log(np.sinh(forward / self.h))

            for i in range(1, n_steps + 1):
                log_sinh_forward[:, i] = log_sinh_forward[:, i - 1] + sigma_sqrt_dt[:, i - 1] \
                 * (ww[:, i - 1] - sigma_sqrt_dt[:, i - 1] / 2 / (1 + np.exp(2 * log_sinh_forward[:, i - 1])))

            sinh_forward_final = np.exp(log_sinh_forward[:, -1])
            forward_final = self.h * np.log(sinh_forward_final + np.sqrt(1 + sinh_forward_final ** 2))
            price = np.mean(np.fmax(cp_sign * (forward_final[:, None] - strike), 0), axis=0) / np.exp(self.intr * texp)

        else:
            forward_h = np.ones((self.n_paths, n_steps + 1)) * forward / self.h
            if self.method == "Euler":
                for i in range(1, n_steps + 1):
                    forward_h[:, i] = forward_h[:, i - 1] + sigma_sqrt_dt[:, i - 1] * np.tanh(forward_h[:, i - 1]) * ww[:, i - 1]
            else:
                for i in range(1, n_steps + 1):
                    forward_h[:, i] = forward_h[:, i - 1] + sigma_sqrt_dt[:, i - 1] * np.tanh(forward_h[:, i - 1]) * \
                     (ww[:, i - 1] + sigma_sqrt_dt[:, i - 1] * (ww[:, i - 1] ** 2 - 1) / 2 / np.cosh(forward_h[:, i - 1]) ** 2)

            forward_final = forward_h[:, -1] * self.h
            price = np.mean(np.fmax(cp_sign * (forward_final[:, None] - strike), 0), axis=0) / np.exp(self.intr * texp)

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
