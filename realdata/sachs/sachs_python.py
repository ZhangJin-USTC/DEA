import os
import pandas as pd
import numpy as np
from abessdag import abessdag
from abessdag_utils import edge_stat,batchimport
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid



# Import real data
sachs = pd.read_csv(os.getcwd()+"\\sachs.csv")
sachs = sachs.drop(sachs.columns[0], axis=1)

# Gold standard by biologists
goldstandard = pd.DataFrame(data=None, columns=["From","To"])
goldstandard.loc[len(goldstandard)]=["Plcg","PIP3"]
goldstandard.loc[len(goldstandard)]=["Plcg","PIP2"]
goldstandard.loc[len(goldstandard)]=["PIP3","PIP2"]
goldstandard.loc[len(goldstandard)]=["Plcg","PKC"]
goldstandard.loc[len(goldstandard)]=["PIP2","PKC"]
goldstandard.loc[len(goldstandard)]=["PIP3","Akt"]
goldstandard.loc[len(goldstandard)]=["PKC","Mek"]
goldstandard.loc[len(goldstandard)]=["PKC","Raf"]
goldstandard.loc[len(goldstandard)]=["PKC","PKA"]
goldstandard.loc[len(goldstandard)]=["PKC","Jnk"]
goldstandard.loc[len(goldstandard)]=["PKC","P38"]
goldstandard.loc[len(goldstandard)]=["PKA","Raf"]
goldstandard.loc[len(goldstandard)]=["PKA","Mek"]
goldstandard.loc[len(goldstandard)]=["PKA","Erk"]
goldstandard.loc[len(goldstandard)]=["PKA","Akt"]
goldstandard.loc[len(goldstandard)]=["PKA","Jnk"]
goldstandard.loc[len(goldstandard)]=["PKA","P38"]
goldstandard.loc[len(goldstandard)]=["Raf","Mek"]
goldstandard.loc[len(goldstandard)]=["Mek","Erk"]
goldstandard.loc[len(goldstandard)]=["Erk","Akt"]

result1 = abessdag(sachs)    # Vanilla ABESS-DAG
edge_stat(goldstandard, result1[1])    # TPR,FDR,SHD,Flip = (30.0, 85.0, 36, 12), 40 edges

kset = pd.DataFrame({"Cause":[],"Effect":[]})
kset = batchimport(kset,["PKA","PKC"],["Mek","Raf","Jnk","P38","Erk","Akt"])

result2 = abessdag(sachs, kset=kset)
edge_stat(goldstandard, result2[1])    # TPR,FDR,SHD,Flip = (65.0, 63.89, 27, 3), 36 edges

# Here are codes of notears by Xun Zheng, which copied from his Github, just for comparison
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

result3 = notears_linear(sachs.values,0.01,'l2')
nz = np.nonzero(result3)
notears_edges = pd.DataFrame({'From':sachs.columns[nz[0]], 'To':sachs.columns[nz[1]]})
edge_stat(goldstandard, notears_edges)    # TPR,FDR,SHD,Flip = (20.0, 69.23, 21, 4), 13 edges

result4 = notears_linear(sachs.values,0.001,'l2')
nz2 = np.nonzero(result4)
notears_edges_2 = pd.DataFrame({'From':sachs.columns[nz2[0]], 'To':sachs.columns[nz2[1]]})
edge_stat(goldstandard, notears_edges_2)     # TPR,FDR,SHD,Flip = (20.0, 76.47, 24, 5), 17 edges