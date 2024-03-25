import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag, sqrtm

def low_block_tri_variable(n, m, Tp1):
    var = (n*Tp1)*[None]
    for t in range(Tp1):
        for i in range(n):
            add_var = cp.Variable((1, m*(t+1)))
            if t == 0 and i == 0:
                var[0] = cp.hstack([add_var, np.zeros((1, m*(Tp1-(t+1))))])
            elif t == Tp1-1:
                var[t*n+i] = add_var
            else:
                var[t*n+i] = cp.hstack([add_var, np.zeros((1, m*(Tp1-(t+1))))])
    var = cp.vstack(var)
    assert var.shape == (n*Tp1, m*Tp1)
    return var

class SLSFinite():
    def __init__(self, A_list, B_list, C_list, D_list, Q_list, R_list):
        """
        Store the variables used for convex optimization in finite time system level synthesis framework.
    
        Parameters
        ----------
        A_list: list of matrices [A_0, ...A_T]
        B_list: list of matrices [B_0, ...B_T]
        C_list: list of matrices [C_0, ...C_T]
            where A_t, B_t, C_t are the matrices in the dynamics of the system at time t.
        
        Attributes
        ----------
        Phi_xx: cvxpy.Variable, shape ((T+1)*nx, (T+1)*nx)
        Phi_xy: cvxpy.Variable, shape ((T+1)*nx, (T+1)*ny)
        Phi_ux: cvxpy.Variable, shape ((T+1)*nu, (T+1)*nx)
        Phi_uy: cvxpy.Variable, shape ((T+1)*nu, (T+1)*ny)
        """
        # init variables
        assert len(A_list) == len(B_list) == len(C_list) == len(D_list)+1
        # define dimanesions
        self.T = len(A_list) - 1
        self.nx = A_list[0].shape[0]
        self.nu = B_list[0].shape[1]
        self.ny = C_list[0].shape[0]

        # define optimization variables
        self.Phi_xx = low_block_tri_variable(self.nx, self.nx, self.T+1)
        self.Phi_uy = low_block_tri_variable(self.nu, self.ny, self.T+1)
        self.Phi_ux = low_block_tri_variable(self.nu, self.nx, self.T+1)
        self.Phi_xy = low_block_tri_variable(self.nx, self.ny, self.T+1)
            
        self.Phi_matrix = cp.bmat( [[self.Phi_xx,   self.Phi_xy], 
                                    [self.Phi_ux,   self.Phi_uy]] )
        # define downshift operator
        self.Z = np.block([ [np.zeros([self.nx,self.T*self.nx]),    np.zeros([self.nx,self.nx])        ],
                            [np.eye(self.T*self.nx),                np.zeros([self.T*self.nx, self.nx])]
                            ])
        
        # define block-diagonal matrices
        self.cal_A = block_diag(*A_list)
        self.cal_B = block_diag(*B_list)
        self.cal_C = block_diag(*C_list)
        self.cal_D = block_diag(*D_list)

        self.cal_Q = block_diag(*Q_list)
        self.cal_R = block_diag(*R_list)

        assert self.Z.shape == self.cal_A.shape
        assert self.Z.shape[0] == self.cal_B.shape[0]
        assert self.Z.shape[0] == self.cal_C.shape[1]

        # dependent variables
        self.F = None
        self.Phi_yx = None
        self.Phi_yy = None
        self.E = None
        self.D = None
        self.F_trunc = None
        self.F_causal_rows_basis = None
        self.Phi_trunc = None
        self.Phi_uy_trunc = None
        self.causal_time = None

    def SLP_constraints(self):
        """
        Compute the system level parametrization constraints used in finite time system level synthesis.

        Return
        ------
        SLP: list of 6 cvxpy.Constraint objects
            These are constraints on the Phi variables consisting of system level parametrization constraints
        """
        Tp1 = self.T + 1
        I = np.eye(Tp1*self.nx)
        SLP = [cp.bmat([[I - self.Z @ self.cal_A, -self.Z @ self.cal_B]]) @ self.Phi_matrix == cp.bmat([[I, np.zeros( (Tp1*self.nx, Tp1*self.ny) )]]),
                self.Phi_matrix @ cp.bmat([[I - self.Z @ self.cal_A], [-self.cal_C]]) == cp.bmat([[I], [np.zeros( (Tp1*self.nu, Tp1*self.nx) )]])]
        return SLP
    
    def L2_constraint(self, gamma, ignore_cal_D=False):
        # return the constraint that the L2 gain (frow w, without x0) must be bounded by gamma

        Phi_xxux = cp.bmat( [[self.Phi_xx], 
                             [self.Phi_ux]] )
        
        if ignore_cal_D:
            constraint_list = [cp.norm( block_diag(sqrtm(self.cal_Q),sqrtm(self.cal_R)) @ Phi_xxux[:,self.nx:] ) <= gamma] # induced 2-norm <= gamma
        else:
            constraint_list = [cp.norm( block_diag(sqrtm(self.cal_Q),sqrtm(self.cal_R)) @ Phi_xxux[:,self.nx:] @ self.cal_D ) <= gamma] # induced 2-norm <= gamma

        return constraint_list
    
    def L2_gain(self, Phi_xx, Phi_ux):
        Phi_xxux = np.bmat( [[Phi_xx], 
                             [Phi_ux]] )
        return np.linalg.norm( block_diag(sqrtm(self.cal_Q),sqrtm(self.cal_R)) @ Phi_xxux[:,self.nx:] @ self.cal_D , 2)

    def sparse_measurements_constraint(self, measurement_times):
        constraints = []
        for t in range(self.T+1):
            if t not in measurement_times:
                for i in range(self.ny):
                    constraints += [ self.Phi_uy[:,t*self.ny+i]==0 ]
        
        return constraints

    def Phi_to_F(self):
        self.F = self.Phi_uy.value - self.Phi_ux.value @ np.linalg.inv((self.Phi_xx.value).astype('float64')) @ self.Phi_xy.value
        return self.F
    
    def robust_l2_gain(self, gamma, epsilon, useRowNorm=False):
        # return gamma_robust = gamma/beta - alpha

        # if a the bound we have on the matrix to factorize is
        #       ||X||_2,row := max_l ||X_{l,:}||_2 <= epsilon
        # then, we need to use
        #       ||X||_2 <= sqrt( (T+1)*n_u )  * ||X||_2,row
        # doing so increase consevatism
        if useRowNorm: 
            epsilon *= np.sqrt( (self.T+1)*self.nu )

        alpha = np.linalg.norm(sqrtm(self.cal_R),2) *epsilon

        temp = np.linalg.norm( (self.Z@self.cal_B).astype('float64') ,2)*epsilon
        # beta = sum_{t=0}^T temp^t
        if temp==0.0:
            beta=self.T+1
        else:
            beta = ( temp**(self.T+1)-1 )/( temp-1 ) # formula for geometric series
        beta *= np.linalg.norm(self.cal_D, 2)

        gamma_robust = gamma/beta - alpha

        print(f"gamma={gamma}, gamma_robust={gamma_robust}, alpha={alpha}, beta={beta}")

        return gamma_robust

    def calculate_dependent_variables(self, key=None):
        """
        Compute the controller F
        """
        F_test = self.Phi_uy.value - self.Phi_ux.value @ np.linalg.inv((self.Phi_xx.value).astype('float64')) @ self.Phi_xy.value
        if key is None:
            return F_test
        
        if key=="Reweighted Nuclear Norm" or key=="Reweighted Sensor Norm":
            self.F = np.linalg.inv( (np.eye(self.nu*(self.T+1)) + self.Phi_ux.value @ self.Z @ self.cal_B).astype('float64') ) @ self.Phi_uy.value
        elif key=="Reweighted Actuator Norm":
            self.F = self.Phi_uy.value @ np.linalg.inv( (np.eye(self.ny*(self.T+1)) + self.cal_C @ self.Phi_xy.value).astype('float64') )
        assert np.all(np.isclose( self.F.astype('float64'), F_test.astype('float64')) )
        filter = np.kron( np.tril(np.ones([self.T+1,self.T+1])) , np.ones([self.nu, self.ny]) )
        self.F = filter*self.F
        return
    
    def F_to_Phi(self,F):
        Phi_xx = np.linalg.inv( (np.eye(self.nx*(self.T+1)) - self.Z @ self.cal_A - self.Z @ self.cal_B @ F @ self.cal_C).astype('float64') )
        Phi_xy = Phi_xx.dot(self.Z).dot(self.cal_B).dot(F)
        Phi_ux = F.dot(self.cal_C).dot(Phi_xx)
        Phi_uy = ( np.eye(self.nu*(self.T+1)) + Phi_ux.dot(self.Z).dot(self.cal_B)).dot(F)
        Phi_uy_sum = F + F.dot(self.cal_C).dot(Phi_xx).dot(self.Z).dot(self.cal_B).dot(F)
        assert np.all(np.isclose( Phi_uy.astype('float64'), Phi_uy_sum.astype('float64')) )
        return Phi_xx, Phi_xy, Phi_ux, Phi_uy
    
    def F_trunc_to_Phi_trunc(self):
        Phi_xx = np.linalg.inv( (np.eye(self.nx*(self.T+1)) - self.Z @ self.cal_A - self.Z @ self.cal_B @ self.F_trunc @ self.cal_C).astype('float64') )
        Phi_xy = Phi_xx.dot(self.Z).dot(self.cal_B).dot(self.F_trunc)
        Phi_ux = self.F_trunc.dot(self.cal_C).dot(Phi_xx)
        Phi_uy = ( np.eye(self.nu*(self.T+1)) + Phi_ux.dot(self.Z).dot(self.cal_B)).dot(self.F_trunc)
        Phi_uy_sum = self.F_trunc + self.F_trunc.dot(self.cal_C).dot(Phi_xx).dot(self.Z).dot(self.cal_B).dot(self.F_trunc)
        assert np.all(np.isclose( Phi_uy.astype('float64'), Phi_uy_sum.astype('float64')) )
        self.Phi_trunc = np.bmat([[Phi_xx, Phi_xy], [Phi_ux, Phi_uy]])
        return
    