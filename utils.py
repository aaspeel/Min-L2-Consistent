import numpy as np
import cvxpy as cp

def periodic_measurement_times(T,period):
    return list( range(0,T+1, period) )

def L2_trajectory(x_list, u_list, w_list, Q, R):
    assert len(w_list)==len(u_list)==len(x_list)-1

    energy_xu=0
    energy_w=0
    for t in range(len(w_list)):
        energy_xu += x_list[t].T @ Q @ x_list[t] + u_list[t].T @ R @ u_list[t]
        energy_w += w_list[t].T @ w_list[t]
    
    if energy_w != 0.0:
        return np.sqrt(energy_xu/energy_w)
    else:
        return 0.0

def causal_factorization(low_btri, rank_eps):
    # low_btri is a block lower triangular matrix, e.g., with (T+1) blocks of the same size (nu,ny)
    
    n_rows = low_btri.shape[0]

    E = np.array([]).reshape((0,low_btri.shape[1]))
    D = np.array([]).reshape((0,0)) # trick
    rank_counter = 0
    rank_low_btri = np.linalg.matrix_rank(low_btri, tol=rank_eps)
    added_rows = rank_low_btri*[None]
    for row in range(n_rows):
        submat_new = low_btri[0:row+1, :] # rows up to row (note: last step row = (T+1)*nu)
        rank_new = np.linalg.matrix_rank(submat_new, tol=rank_eps)
        if rank_new == rank_counter + 1:
            added_rows[rank_counter] = row
            rank_counter += 1
            # add vector to E matrix
            E = np.vstack([E, submat_new[row:row+1, :]])
            # modify D matrix
            unit = np.zeros([1, rank_counter]) 
            unit[0, -1] = 1. # add a 1 at the last column of unit
            D = np.hstack([D, np.zeros([row, 1])]) # D is (row, rank_counter)
            D = np.vstack([D, unit]) # D is (row+1, rank_counter)
            assert E.shape == (rank_counter, low_btri.shape[1])
            assert D.shape == (row+1, rank_counter)
        elif rank_new == rank_counter:
            # solve linear system
            c = np.linalg.lstsq( E.T , low_btri[row, :], rcond=None)[0]
            c = c.reshape(([1, rank_counter]))
            D = np.vstack([D, c])
            assert E.shape == (rank_counter, low_btri.shape[1])
            assert D.shape == (row+1, rank_counter)
        else:
            raise Exception('Rank increased more than 1.')

    assert E.shape == (rank_counter, low_btri.shape[1])
    assert D.shape == (low_btri.shape[0], rank_counter)
    assert rank_counter == rank_low_btri
    assert len(added_rows) == rank_low_btri
    return D,E


def approximate_causal_factorization(X, tol, useRowNorm=False):
    if useRowNorm:
        return approximate_causal_factorization_row_norm(X, tol)
    else:
        return approximate_causal_factorization_2_norm(X, tol)

def approximate_causal_factorization_2_norm(X, tol):
    # return D and E s.t. ||X-DE||_2 <= tol where || Z ||_2 is the spectral norm of the matrix Z.
    (n_rows,n_cols) = np.shape(X)

    D = np.zeros((0,0))
    E = np.zeros((0,n_cols))
    r = 0 # band of the factorization
    encoded_rows = []
    for l in range(n_rows):

        # compute the error
        if r==0:
            err = np.linalg.norm( X[0:l+1,:] ,2)
            d = np.zeros((1,r))
        else:
            d = cp.Variable((1,r))
            objective = cp.Minimize( cp.norm( X[0:l+1,:] - cp.vstack([D, d])@E , 2) )
            problem = cp.Problem(objective,[])
            err = problem.solve(solver=cp.MOSEK,
                                mosek_params = {'MSK_DPAR_INTPNT_CO_TOL_DFEAS':  1e-11,
                                                },
                                verbose=False)
            if problem.status != cp.OPTIMAL:
                raise Exception("Solver did not converge!")
            d = d.value
        if err <= tol:
            D = np.vstack([D, d])
        else:
            E = np.vstack([E, X[l,:]])
            D = np.block( [[D,               np.zeros((l,1))],
                          [np.zeros((1,r)), np.array([[1]])]])
            r = r+1
            encoded_rows.append(l)
    assert np.linalg.norm( X-D@E , 2) <= tol
    return D,E,encoded_rows

def approximate_causal_factorization_row_norm(X, tol):
    # return D and E s.t. ||X-DE||_{r,2} <= tol, where || Z ||_{r,2} := max_l || Z[l,:] ||_2, i.e., it is the max of the 2-norm of each row
    tol2 = tol**2
    (n_rows,n_cols) = np.shape(X)

    D = np.zeros((0,0))
    E = np.zeros((0,n_cols))
    r = 0 # band of the factorization
    for l in range(n_rows):
        d,err2,_,_ = np.linalg.lstsq(E.T, (X[l,:]).T , rcond=None) # err2 is a (1,) vector that contains the squared norm of the residual
        if err2[0] <= tol2:
            D = np.vstack([D, d])
        else:
            E = np.vstack([E, X[l,:]])
            D = np.block( [[D,               np.zeros((l,1))],
                          [np.zeros((1,r)), np.array([[1]])]])
            r = r+1
    assert np.max(np.linalg.norm( X-D@E , axis=1 )) <= tol
    return D,E