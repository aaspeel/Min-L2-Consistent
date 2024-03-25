import numpy as np
from scipy.linalg import block_diag

def minimax(A, B, D, Q, T, gamma, measurement_times=None, w_list=None):
    nx = A.shape[0]; nu = B.shape[1]; nw = D.shape[1]

    # Backward loop
    N_list = (T+1)*[np.zeros([nx,nx])]
    Theta_list = (T+1)*[np.zeros([nx,nx])]
    Lambda_list = T*[np.zeros([nx,nx])]
    Lambda_inv_list = T*[np.zeros([nx,nx])] 
    V_list = T*[np.zeros([nx,nx])]
    V_inv_list = T*[np.zeros([nx,nx])]
    S_list = T*[np.zeros([nw,nx])]
    L_list = T*[np.zeros([nw,nx])]
    R_list = T*[np.zeros([nu,nx])]
    for t in range(T-1,0-1,-1): # T-1 , T-1, ..., 1, 0
        # Lambda and N must be in the same loop since they depend from each others
        Lambda_list[t] = np.eye(nx) + ( B@B.T - gamma**(-2)*D@D.T ) @ N_list[t+1]
        Lambda_inv_list[t] = np.linalg.inv( Lambda_list[t] )
        N_list[t] = Q + A.T @ N_list[t+1] @ Lambda_inv_list[t] @ A

        # R is computed from N and Lambda
        R_list[t] = -B.T @ N_list[t+1] @ Lambda_inv_list[t] @ A

        # L is computed from N and Lambda
        L_list[t] = gamma**(-2)*D.T@N_list[t+1]@Lambda_inv_list[t]@A

        # V and Theta must be in the same loop since they depend from each others (and on N)
        V_list[t] = np.eye(nx) - gamma**(-2)*D@D.T@Theta_list[t+1]
        V_inv_list[t] = np.linalg.inv( V_list[t] )
        if t in measurement_times:
            Theta_list[t] = N_list[t]
        else:
            Theta_list[t] = Q + A.T@Theta_list[t+1]@V_inv_list[t]@A

        # S is computed from Theta and V
        S_list[t] = gamma**(-2)*D.T@Theta_list[t+1]@V_inv_list[t]@A

    # Forward loop
    x_list = (T+1)*[np.zeros(nx)] # x(t)
    x_pred_list = (T+1)*[np.zeros(nx)] # x(t|t-1)
    x_corr_list = T*[np.zeros(nx)] # x(t|t)
    if w_list is None:
        w_list = T*[np.zeros(nx)]
    u_list = T*[np.zeros(nx)]
    for t in range(0,T):
        if t in measurement_times:
            x_corr_list[t] = x_list[t]
        else:
            x_corr_list[t] = x_pred_list[t]
        
        u_list[t] = R_list[t]@x_corr_list[t]

        if w_list is None:
            w_list[t] = S_list[t]@x_list[t] + (L_list[t] - S_list[t])@x_corr_list[t]

        x_pred_list[t+1] = Lambda_inv_list[t]@A@x_pred_list[t]
        x_list[t+1] = A@x_list[t] + B@u_list[t] + D@w_list[t]

    return x_list,u_list


def compute_transmission_times(A, B, D, Q, T, gamma):
    nx = A.shape[0]; nu = B.shape[1]; nw = D.shape[1]

    N_list = (T+1)*[np.zeros([nx,nx])]
    Lambda_list = T*[np.zeros([nx,nx])]
    Lambda_inv_list = T*[np.zeros([nx,nx])]
    for t in range(T-1,0-1,-1): # T-1 , T-1, ..., 1, 0
        # Lambda and N must be in the same loop since they depend from each others
        Lambda_list[t] = np.eye(nx) + ( B@B.T - gamma**(-2)*D@D.T ) @ N_list[t+1]
        Lambda_inv_list[t] = np.linalg.inv( Lambda_list[t].astype('float64') )
        N_list[t] = Q + A.T @ N_list[t+1] @ Lambda_inv_list[t] @ A

    transmission_times = [0]
    last_transmission_time = 0
    r = 1
    while last_transmission_time + r <= T:
        #compute M
        D_bar = np.zeros([nx*r,nw*r])
        block_rows = np.zeros([nx,nw*r])
        for k in range(r):
            block_rows = A @ block_rows
            block_rows[:,k*nw:(k+1)*nw] = D # [A^k*D A^(k-1)*D ... AD A]
            D_bar[k*nx:(k+1)*nx,:] = block_rows
        if r==1:
            N_bar = N_list[last_transmission_time+1]
        else:
            N_bar = block_diag( np.kron(np.eye(r-1),Q) , N_list[last_transmission_time+r] )
        M = gamma**2*np.eye(r*nw) - D_bar.T @ N_bar @ D_bar

        if np.all(np.linalg.eigvals(M) > 0): # M is postitve definite
            r=r+1
        else:
            assert r!=1 # this case should not happen. If it happens, this assert avoids an infinite loop.
            last_transmission_time += (r-1)
            transmission_times.append(last_transmission_time)
            r=1
    
    # repeat each transmission time nx times
    transmission_times = [ele for ele in transmission_times for i in range(nx)]

    return transmission_times

        
