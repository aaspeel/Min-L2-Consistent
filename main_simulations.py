import numpy as np
import warnings

from load_systems import *
from SLSFinite import *
from optimize_SLS import *
from minimax import *
from plots import *
import time
import pickle

np.random.seed(1)

file_name = 'simulation_results/variables.pkl'

A, B, C, D, Q, R, T = load_drone() # load_scalar_system() OR load_drone()
A_list = (T+1)*[A]; B_list = (T+1)*[B]; C_list = (T+1)*[C]; D_list = T*[D]; Q_list = (T+1)*[Q]; R_list = (T+1)*[R]
SLS_data = SLSFinite(A_list, B_list, C_list, D_list, Q_list, R_list)

opt_eps = 1e-11
truncation_tol = 1e-8
N_nuclear_norm = 8

### PERIODIC GAMMA
print("\n============================================")
print("============== PERIODIC GAMMA ==============")
period = 3

gamma_periodic, prob = min_gamma(SLS_data, period, gamma=None, opt_eps=opt_eps)
print(f"The optimal gamma for a period {period} is: {gamma_periodic}")
transmission_times_periodic = periodic_measurement_times(T, period)
transmission_times_periodic = [ele for ele in transmission_times_periodic for i in range(SLS_data.nx)] # repeat each transmission time nx times
n_communications_periodic = np.size(transmission_times_periodic)
print("Periodic communication times: ", transmission_times_periodic)
print(f"Periodic communication gives {n_communications_periodic} scalar communications")

gamma_bound = gamma_periodic

### PSD METHOD (for comparison)
print("\n============================================")
print("================ PSD METHOD ================")
duration_psd = time.time()
transmission_times_psd = compute_transmission_times(A, B, D, Q, T, gamma_bound)
duration_psd = time.time() - duration_psd
n_transmission_times_psd = np.size(transmission_times_psd)
print("PSD transmission times: ", transmission_times_psd)
print(f"PSD method gives {n_transmission_times_psd} scalar communications")

### APPROXIMATE CAUSAL FACTORIZATION
print("\n============================================")
print("===== APPROXIMATE CAUSAL FACTORIZATION =====")

duration_optimization = time.time()
Phi_xx, _, Phi_ux, _, prob_rank = min_rank(SLS_data, gamma_bound, truncation_tol=truncation_tol, N=N_nuclear_norm)
duration_optimization = time.time() - duration_optimization

duration_factorization = time.time()
Dec_Phi_ux, Enc_Phi_ux, encoded_rows = approximate_causal_factorization(Phi_ux, truncation_tol)
duration_factorization = time.time() - duration_factorization

transmission_times_rank = [l//SLS_data.nu for l in encoded_rows]
print("transmission_times_rank")
print(transmission_times_rank)

Dec_F = Dec_Phi_ux
Enc_F = Enc_Phi_ux@np.linalg.inv(Phi_xx.astype('float64'))

n_transmission_times_rank = np.shape(Enc_Phi_ux)[0]
F_trunc = Dec_F@Enc_F

Phi_xx_trunc, _, Phi_ux_trunc, _ = SLS_data.F_to_Phi(F_trunc)
L2_trunc = SLS_data.L2_gain(Phi_xx_trunc, Phi_ux_trunc)
print("L2-gain for F_trunc is L2_trunc=",L2_trunc, "while gamma_bound is", gamma_bound)
if L2_trunc <= gamma_bound+1e-9:
    print("FEASIBLE: The truncated F satisfies the L2-gain !")
else:
    warnings.warn("INFEASIBILE: The truncated F does not satisfy the L2-gain !")
print(f"Approx. Causal Factorization gives {n_transmission_times_rank} transmissions")

### EPSILON VS NUMBER OF TRANSMISSIONS
print("\n============================================")
print("================ EPSILON vs BAND ===========")

list_truncation_tol = [1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-12, 0]
list_n_transmission_trunc_tol = []
for i in range(len(list_truncation_tol)):
    epsilon=list_truncation_tol[i]
    try:
        Phi_xx, _, Phi_ux, _, prob_rank = min_rank(SLS_data, gamma_bound, truncation_tol=epsilon, N=N_nuclear_norm)
        _, _, encoded_rows = approximate_causal_factorization(Phi_ux, epsilon)
        list_n_transmission_trunc_tol.append(len(encoded_rows))
    except:
        list_n_transmission_trunc_tol.append(None)
        print("Rank minimization problem was infeasible for epsilon=",epsilon)

print("list_truncation_tol:  ", list_truncation_tol)
print("list_n_transmissions: ", list_n_transmission_trunc_tol)

### GAMMA VS NUMBER OF TRANSMISSIONS
print("\n============================================")
print("================ GAMMA vs BAND ===========")

list_gamma = gamma_bound + np.linspace(-0.5,2.5,10+1) # include 0 in the linspace
list_n_transmission_rank_gamma = []
list_n_transmission_PSD_gamma = []
for i in range(len(list_gamma)):
    try:
        Phi_xx, _, Phi_ux, _, prob_rank = min_rank(SLS_data, list_gamma[i], truncation_tol=truncation_tol, N=N_nuclear_norm)
        _, _, encoded_rows = approximate_causal_factorization(Phi_ux, truncation_tol)
        list_n_transmission_rank_gamma.append(len(encoded_rows))
    except:
        list_n_transmission_rank_gamma.append(None)
        print("Rank minimization problem was infeasible for gamma=",list_gamma[i])

    try:
        measurement_times = compute_transmission_times(A, B, D, Q, T, list_gamma[i])
        list_n_transmission_PSD_gamma.append(np.size(measurement_times))

    except:
        list_n_transmission_PSD_gamma.append(None)
        print("PSD method fails for gamma=",list_gamma[i])

print("list_gamma:           ", list_gamma)
print("list_n_transmissions_rank: ", list_n_transmission_rank_gamma)
print("list_n_transmissions_PSD: ", list_n_transmission_PSD_gamma)

### SAVE VARIABLES ###
variables_to_save = {
    'A': A,
    'B': B,
    'C': C,
    'D': D,
    'Q': Q,
    'R': R,
    'T': T,
    'A_list': A_list,
    'B_list': B_list,
    'C_list': C_list,
    'D_list': D_list,
    'Q_list': Q_list,
    'R_list': R_list,
    'SLS_data': SLS_data,
    'opt_eps': opt_eps,
    'truncation_tol': truncation_tol,
    'N_nuclear_norm': N_nuclear_norm,
    'period': period,
    'gamma_periodic': gamma_periodic,
    'prob': prob,
    'transmission_times_periodic': transmission_times_periodic,
    'n_communications_periodic': n_communications_periodic,
    'gamma_bound': gamma_bound,
    'duration_optimization': duration_optimization,
    'duration_factorization': duration_factorization,
    'Phi_xx': Phi_xx,
    'Phi_ux': Phi_ux,
    'Enc_Phi_ux': Enc_Phi_ux,
    'Dec_Phi_ux': Dec_Phi_ux,
    'Dec_F': Dec_F,
    'Enc_F': Enc_F,
    'encoded_rows': encoded_rows,
    'transmission_times_rank': transmission_times_rank,
    'n_transmission_times_rank': n_transmission_times_rank,
    'F_trunc': F_trunc,
    'Phi_xx_trunc': Phi_xx_trunc,
    'Phi_ux_trunc': Phi_ux_trunc,
    'L2_trunc': L2_trunc,
    'duration_psd': duration_psd,
    'transmission_times_psd': transmission_times_psd,
    'n_transmission_times_psd': n_transmission_times_psd,
    'list_truncation_tol': list_truncation_tol,
    'list_n_transmission_trunc_tol': list_n_transmission_trunc_tol,
    'list_gamma': list_gamma,
    'list_n_transmission_rank_gamma': list_n_transmission_rank_gamma,
    'list_n_transmission_PSD_gamma': list_n_transmission_PSD_gamma
}

with open(file_name, 'wb') as f:
    pickle.dump(variables_to_save, f)
print("Results have been save in the file:", file_name)