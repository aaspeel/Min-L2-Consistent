import pickle
from plots import *

save = True

file_name = 'simulation_results/variables.pkl'

# Load the data from the pickle file
with open(file_name, 'rb') as f:
    loaded_variables = pickle.load(f)

# Retrieve the variables
A = loaded_variables['A']
B = loaded_variables['B']
C = loaded_variables['C']
D = loaded_variables['D']
Q = loaded_variables['Q']
R = loaded_variables['R']
T = loaded_variables['T']
A_list = loaded_variables['A_list']
B_list = loaded_variables['B_list']
C_list = loaded_variables['C_list']
D_list = loaded_variables['D_list']
Q_list = loaded_variables['Q_list']
R_list = loaded_variables['R_list']
SLS_data = loaded_variables['SLS_data']
opt_eps = loaded_variables['opt_eps']
truncation_tol = loaded_variables['truncation_tol']
N_nuclear_norm = loaded_variables['N_nuclear_norm']
period = loaded_variables['period']
gamma_periodic = loaded_variables['gamma_periodic']
prob = loaded_variables['prob']
transmission_times_periodic = loaded_variables['transmission_times_periodic']
n_communications_periodic = loaded_variables['n_communications_periodic']
gamma_bound = loaded_variables['gamma_bound']
duration_optimization = loaded_variables['duration_optimization']
duration_factorization = loaded_variables['duration_factorization']
Phi_xx = loaded_variables['Phi_xx']
Phi_ux = loaded_variables['Phi_ux']
Enc_Phi_ux = loaded_variables['Enc_Phi_ux']
Dec_Phi_ux = loaded_variables['Dec_Phi_ux']
Dec_F = loaded_variables['Dec_F']
Enc_F = loaded_variables['Enc_F']
encoded_rows = loaded_variables['encoded_rows']
transmission_times_rank = loaded_variables['transmission_times_rank']
n_transmission_times_rank = loaded_variables['n_transmission_times_rank']
F_trunc = loaded_variables['F_trunc']
Phi_xx_trunc = loaded_variables['Phi_xx_trunc']
Phi_ux_trunc = loaded_variables['Phi_ux_trunc']
L2_trunc = loaded_variables['L2_trunc']
duration_psd = loaded_variables['duration_psd']
transmission_times_psd = loaded_variables['transmission_times_psd']
n_transmission_times_psd = loaded_variables['n_transmission_times_psd']
list_truncation_tol = loaded_variables['list_truncation_tol']
list_n_transmission_trunc_tol = loaded_variables['list_n_transmission_trunc_tol']
list_gamma = loaded_variables['list_gamma']
list_n_transmission_rank_gamma = loaded_variables['list_n_transmission_rank_gamma']
list_n_transmission_PSD_gamma = loaded_variables['list_n_transmission_PSD_gamma']

### PLOTS AND PRINTS ###
print("gamma_bound=",gamma_bound, "truncation_tol=",truncation_tol, "N_nuclear_norm=",N_nuclear_norm, "period=",period)

print("Duration rank minimization =", duration_optimization, "seconds")
print("Duration factorization =", duration_factorization, "seconds")
print("Duration total rank method =", duration_optimization+duration_factorization, "seconds")
print("Duration PSD method =", duration_psd, "seconds")

fig_causal_factorization = plot_causal_factorization(SLS_data, F_trunc, Dec_F, Enc_F)

fig_transmission_times = plt.figure()
plot_transmission_times(fig_transmission_times, transmission_times_periodic, SLS_data.T, color='black', linestyle='dashdot', label="Periodic")
plot_transmission_times(fig_transmission_times, transmission_times_rank, SLS_data.T, color='red', linestyle='solid', label="Rank (ours)")
plot_transmission_times(fig_transmission_times, transmission_times_psd, SLS_data.T, color='blue', linestyle='dashed', label="Minimax")

fig_transmissions_vs_tol = plot_transmissions_vs_tol(list_truncation_tol, list_n_transmission_trunc_tol, color='red', marker='o')

fig_transmissions_vs_gamma = plt.figure()
plot_transmissions_vs_gamma(fig_transmissions_vs_gamma, list_gamma, list_n_transmission_rank_gamma, color='red', marker='o', label="Rank (ours)")
plot_transmissions_vs_gamma(fig_transmissions_vs_gamma, list_gamma, list_n_transmission_PSD_gamma, color='blue', marker='s', label="Minimax")

if save:
    fig_causal_factorization.savefig("simulation_results/causal_factorization.pdf", bbox_inches="tight")
    fig_transmission_times.savefig("simulation_results/transmission_times.pdf", bbox_inches="tight")
    fig_transmissions_vs_gamma.savefig("simulation_results/transmissions_vs_gamma.pdf", bbox_inches="tight")
    fig_transmissions_vs_tol.savefig("simulation_results/transmissions_vs_tol.pdf", bbox_inches="tight")

plt.show()