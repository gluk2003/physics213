#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 22:49:26 2023

@author: golikovalex
"""
from scipy.integrate import quad
import json
import matplotlib.pyplot as plt
import numpy as np
import opt


if __name__ == "__main__":
    with open('jla_mub.txt') as file:
        data = np.loadtxt(file)
        z, mu = data[:, 0], data[:, 1]
        x0 = [50, 0.5]
        c = 3 * 10**11
        
        def f(H, omega):
            d = []
            func = lambda x: 1 / np.sqrt((1 - omega) * (1 + x) ** 3 + omega)
            for z1 in z:
                intg = quad(func, 0, z1)[0]
                d.append(c / H * (1 + z1) * intg)
            d = np.asarray(d)
            mu = 5 * np.log10(d) - 5
            return mu
        
        def j(H, omega):
            d = []
            mu_omega = []
            func = lambda x: 1 / np.sqrt((1 - omega) * (1 + x) ** 3 + omega)
            func1 = lambda x: 0.5 * (-1 + (x + 1)**3) / np.sqrt((1 - omega) * (1 + x) ** 3 + omega)**3
            for z1 in z:
                intg = quad(func, 0, z1)[0]
                d.append((c / H * (1 + z1) * intg))
                intg = quad(func1, 0, z1)[0]
                mu_omega.append(c / H * (1 + z1) * intg)
            d = np.asarray(d)
            mu_H = -5 / H / np.log(10) * np.ones(d.size)
            mu_omega = 5 / d / np.log(10) * np.asarray(mu_omega)
            res = np.hstack((mu_H.reshape(-1, 1), mu_omega.reshape(-1, 1)))
            return res
        
        nfev_gs, cost_gs, g_gs, x_gs = opt.gauss_newton(mu, f, j, x0)
        nfev_lm, cost_lm, g_lm, x_lm = opt.lm(mu, f, j, x0)
        
        #results
        res ={
            "Gauss-Newton": {"H0": x_gs[0].round(), "Omega": x_gs[1].round(2), "nfev": nfev_gs},
            "Levenberg-Marquardt": {"H0": x_lm[0].round(), "Omega": x_lm[1].round(2), "nfev": nfev_lm}
            }   
            
        with open('parameters.json', 'w') as file:
            json.dump(res, file, indent=2)
        
        
        #graphs
        plt.plot(z, mu, 'P', label="experimental")
        plt.plot(z, f(*x_gs), label='gauss-newton')
        plt.plot(z, f(*x_lm), label='levenberg-marquardt')
        plt.xlabel("z")
        plt.ylabel("μ")
        plt.grid()
        plt.legend()
        plt.savefig('mu-z.png')
        
        plt.figure()
        plt.plot(np.arange(nfev_gs), cost_gs, label='gauss-newton')
        plt.plot(np.arange(nfev_lm + 1), cost_lm, label='levenberg-marquardt')
        plt.xlabel("iter step")
        plt.ylabel("error value")
        plt.grid()
        plt.legend()
        plt.savefig('cost.png')
        
        
                
            
                
    