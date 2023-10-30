#!/usr/bin/env python3

from collections import namedtuple
import numpy as np


Result = namedtuple('Result', ('nfev', 'cost', 'gradnorm', 'x'))
Result.__doc__ = """Результаты оптимизации

Attributes
----------
nfev : int
    Полное число вызовов модельной функции
cost : 1-d array
    Значения функции потерь 0.5 sum(y - f)^2 на каждом итерационном шаге.
    В случае метода Гаусса—Ньютона длина массива равна nfev, в случае ЛМ-метода
    длина массива — менее nfev
gradnorm : float
    Норма градиента на финальном итерационном шаге
x : 1-d array
    Финальное значение вектора, минимизирующего функцию потерь
"""


def gauss_newton(y, f, j, x0, k=1, tol=1e-4):
    x = np.asarray(x0)
    nfev = 0
    cost = []
    while True:
        J = j(*x)
        dif = f(*x) - y
        grad = J.T @ dif
        grad_norm = np.linalg.norm(grad)
        dx = np.linalg.solve(J.T @ J, -grad)
        x = x + k * dx
        cost.append(0.5 * np.dot(dif, dif))
        nfev += 1
        
        if len(cost) >= 2 and np.abs(cost[-1] - cost[-2]) <= tol * np.abs(cost[-1]):
            break
        
        if grad_norm < tol * np.abs(cost[-1]) / np.linalg.norm(x):
            break
        
    return Result(nfev=nfev,
                  cost=np.asarray(cost),
                  gradnorm=grad_norm,
                  x=x)

def Fi(y, J, grad, x, lmbd, f):
    dx = np.linalg.solve(J.T @  J + np.identity(J.shape[1]) * lmbd, -grad)
    x = x + dx
    F = 0.5 * np.dot(f(*x) - y, f(*x) - y)
    return x, F
    
    
def lm(y, f, j, x0, lmbd0=1e-2, nu=2, tol=1e-4):
    x = np.asarray(x0)
    nfev = 0
    J = j(*x)
    dif = f(*x) - y
    F = 0.5 * np.dot(dif, dif)
    cost= [F]
    grad = J.T @ dif
    lmbd = lmbd0
    while True:
        lmbd0 = lmbd
        lmbd1 = lmbd / nu
        
        # решим для lmbd0
        x0, F0 = Fi(y, J, grad, x, lmbd0, f)
        
        # решим для lmbd1
        x1, F1 = Fi(y, J, grad, x, lmbd1, f)
        
        
        if F1 <= F:
            lmbd = lmbd1
        elif F1 > F and F0 <= F:
            lmbd = lmbd0
        else:
            while F > Fi(y, J, grad, x, lmbd, f)[1]:
                lmbd = lmbd * nu
            
        x, F = Fi(y, J, grad, x, lmbd, f)
        dif = f(*x) - y
        grad = J.T @ dif
        
        grad_norm = np.linalg.norm(grad)
        cost.append(F)
                
            
        nfev += 1
        if len(cost) >= 2 and np.abs(cost[-1] - cost[-2]) <= tol * np.abs(cost[-1]):
            break
        
        if grad_norm < tol * np.abs(cost[-1]) / np.linalg.norm(x):
            break
        
    return Result(nfev=nfev,
                  cost=np.asarray(cost),
                  gradnorm=grad_norm,
                  x=x)
        
        
