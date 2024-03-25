import numpy as np
import math
import scipy as sp

# x_{t+1} = A_t*x_t + B_t*u_t + D_t*w_t
# y_t = C*x_t + v_t
# sum_t x_t'*Q*x_t + u_t'*R*u_t <= gamma^2 sum_t w_t'*w_t 

def load_drone():
    # double integrator
    T = 20
    dt = 0.1
    A_0 = np.block([[np.zeros([2,2]), np.eye(2)], [np.zeros([2,2]), np.zeros([2,2])]])
    B_0 = np.block([[np.zeros([2,2])],[np.eye(2)]])
    A = sp.linalg.expm(A_0*dt)
    B = np.sum([np.linalg.matrix_power(A_0*dt,i)/math.factorial(i+1) for i in np.arange(100)], axis=0).dot(B_0)
    C = np.eye(4)
    D = np.eye(4)

    Q = np.eye(4)
    R = np.eye(2) 
    return A, B, C, D, Q, R, T

def load_scalar_system():
    T = 37
    A = np.array([[1]])
    B = np.array([[1]])
    C = np.array([[1]])
    D = np.eye(1)

    Q = 0.01*np.eye(1)
    R = np.eye(1)
    return A, B, C, D, Q, R, T