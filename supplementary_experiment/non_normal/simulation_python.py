import os
import pandas as pd
import numpy as np
from abessdag import abessdag
from abessdag_utils import edge_stat
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
import time

def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


for err in ["Uniform","Laplace","Gumbel","t"]:
    for y in [30,100,300]:
        result1 = pd.DataFrame(data=None, columns=["Time","TPR","FDR","SHD"])
        result2 = pd.DataFrame(data=None, columns=["Time","TPR","FDR","SHD"])
        for k in range(100):
            pth = os.getcwd()+r"\data\ER2\{err}\n{y}\ER2_{err}_p30_n{y}_{k}.csv".format(err=err,y=y,k=k)
            data = pd.read_csv(pth)
            data = data.iloc[: , 1:]
            pth_info = os.getcwd()+r"\data\ER2\graph\ER2_p30_{k}.csv".format(k=k)
            ginfo = pd.read_csv(pth_info)
            start = time.perf_counter()
            w = abessdag(data)
            end = time.perf_counter()
            time1 = round(end-start,2)
            ss1 = edge_stat(ginfo,w[1])
            result1.loc[len(result1)] = [time1,ss1[0],ss1[1],ss1[2]]


            start = time.perf_counter()
            w2 = notears_linear(data.values,0.01,'l2')
            end = time.perf_counter()
            time2 = round(end-start,2)
            nz = np.nonzero(w2)
            notears_edges = pd.DataFrame({'From':data.columns[nz[0]], 'To':data.columns[nz[1]]})
            ss2 = edge_stat(ginfo, notears_edges)  
            result2.loc[len(result2)] = [time2,ss2[0],ss2[1],ss2[2]]

            pth_out = os.getcwd()+r"\data\ER2\{err}\n{y}".format(err=err,y=y)
            result1.to_csv(pth_out+r"\Result_abessdag.csv")
            result2.to_csv(pth_out+r"\Result_notears.csv")
