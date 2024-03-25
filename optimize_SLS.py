import numpy as np
import cvxpy as cp
from SLSFinite import *
import matplotlib.pyplot as plt
from utils import *

verbose=False

def min_gamma(SLS_data, period=1, gamma=None, opt_eps=1e-11):
    """
    Parameters
    ----------
    A_list: list of matrices [A_0, ..., A_T]
    B_list: list of tensors [B_0, ..., B_T]
    C_list: list of tensors [C_0, ..., C_T]
        where A_t, B_t, C_t are the matrices in the dynamics of the system at time t.

    Returns
    -------
    result: float
        Optimal cost (np.inf if problem is not feasible).
    SLS_data: SLSFinite object
        Instance of the class SLSFinite containing the variables corresponding
        to the optimal cost.
    """

    # constraints
    constraints = []
    constraints += SLS_data.SLP_constraints()
    if gamma is None:
        gamma = cp.Variable(1)
    constraints += SLS_data.L2_constraint(gamma)

    measurement_times = periodic_measurement_times(SLS_data.T,period)
    constraints += SLS_data.sparse_measurements_constraint(measurement_times)

    # objective function
    objective = cp.Minimize(gamma)
    problem = cp.Problem(objective, constraints)

    # solve
    problem.solve(solver=cp.MOSEK,
                    mosek_params = {'MSK_DPAR_INTPNT_CO_TOL_DFEAS':  opt_eps,
                                        },
                    verbose=False)
    if problem.status != cp.OPTIMAL:
        raise Exception("Solver did not converge!")
    
    return problem.value, problem

def min_rank(SLS_data, gamma, truncation_tol=None, N=4, delta=0.01, opt_eps=1e-11):
    """
    Parameters
    ----------
    A_list: list of matrices [A_0, ...A_T]
    B_list: list of tensors [B_0, ...B_T]
    C_list: list of tensors [C_0, ...C_T]
        where A_t, B_t, C_t are the matrices in the dynamics of the system at time t.

    Returns
    -------
    result: float
        Optimal cost (np.inf if problem is not feasible).
    SLS_data: SLSFinite object
        Instance of the class SLSFinite containing the variables corresponding
        to the optimal cost.
    Lambda: cvxpy.Variable, shape (H_x[0], H_w[0])
        Polytope containment variable corresponding to the optimal cost.
    """

    # constraints
    constraints = []
    constraints += SLS_data.SLP_constraints()
    if truncation_tol is not None: # truncation will be performed and a robust gamma is used
        gamma = SLS_data.robust_l2_gain(gamma, truncation_tol) # compute a gamma robust to a truncation of Phi_u up to 'truncation_tol'
        constraints += SLS_data.L2_constraint(gamma , ignore_cal_D=True)
    else:
        constraints += SLS_data.L2_constraint(gamma , ignore_cal_D=False)

    # Initialize Paramters
    W_1 = cp.Parameter(2*[SLS_data.nu*(SLS_data.T+1)])
    W_2 = cp.Parameter(2*[SLS_data.ny*(SLS_data.T+1)])
    W_1.value = delta**(-1/2)*np.eye(SLS_data.nu*(SLS_data.T+1))
    W_2.value = delta**(-1/2)*np.eye(SLS_data.ny*(SLS_data.T+1))

    # objective function
    objective = cp.Minimize(cp.norm(W_1 @ SLS_data.Phi_uy @ W_2, 'nuc'))

    # define problem
    problem = cp.Problem(objective, constraints)

    result_list = N*[None]
    # nuclear norm reweighting algorithm
    for k in range(N):
        print("Reweighting itereation",k+1,"/",N)
        # solve
        result = problem.solve(solver=cp.MOSEK,
                               mosek_params = {'MSK_DPAR_INTPNT_CO_TOL_DFEAS':  opt_eps,
                                            },
                               verbose=False)
        if problem.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")
        result_list[k] = result

        #update params (reweighting)
        [U, S, Vh] = np.linalg.svd((W_1 @ SLS_data.Phi_uy @ W_2).value , full_matrices=False)

        Y = np.linalg.inv(W_1.value).dot((W_1 @ SLS_data.Phi_uy @ W_2).value).dot(Vh.T).dot(U.T).dot(np.linalg.inv(W_1.value))
        # Y = W_1^-1 * W1 * Phi_uy * W_2 * Vh' * U'* W_1^-1 
        Z = np.linalg.inv(W_2.value).dot(Vh.T).dot(U.T).dot((W_1 @ SLS_data.Phi_uy @ W_2).value).dot(np.linalg.inv(W_2.value))
        # Z = W_2^-1 * Vh' * U' * W_1 * Phi_uy * W_2 * W_2^-1

        # auxiliary function
        def update_W(Q, dim, delta):
            W = (Q + delta*np.eye(dim))
            [eig, eigv] = np.linalg.eigh(W)
            #assert np.all(eig > 0)
            W = eigv.dot(np.diag(eig**(-1/2))).dot(np.linalg.inv(eigv))
            return W
        
        W_1.value = update_W(Y, SLS_data.nu*(SLS_data.T+1), delta)
        W_2.value = update_W(Z, SLS_data.ny*(SLS_data.T+1), delta)
    
    return SLS_data.Phi_xx.value, SLS_data.Phi_xy.value, SLS_data.Phi_ux.value, SLS_data.Phi_uy.value, problem
