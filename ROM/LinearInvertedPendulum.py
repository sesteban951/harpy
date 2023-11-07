import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pydrake.trajectories import BezierCurve

# LIP class for generating trajectories
class LinearInvertedPendulum():
    """
    Simple model of a planar linear inverted pendulum (LIP).
    Inputs: N (number of steps) as integer
            x0 (initial state)  as numpy array
    Outputs: t (time), x (state), b (foot placement) trajectory as time series numpy array
    """    
    def __init__(self,N,x0):

        # default class parameters
        self.g  = 9.81                        # gravity [m/s^2]
        self.z0 = 0.5                         # const height [m]
        lam = np.sqrt(self.g/self.z0)         # natural frequency [1/s]
        self.A = np.array([[0,1],[lam**2,0]]) # drift matrix of LIP

        self.z_apex = self.z0*0.25            # apex height [m]
        self.T = 0.4                          # step period [s]
        
        self.alpha = 0.28                     # raibert controller parameters
        self.beta = 1                         # raibert controller parameters
        
        self.N = N                            # number of steps
        self.dt = 0.02                        # time step [s]
        self.x0 = x0                          # initial state [m, m/s]

        # checks that arguments are valid
        assert isinstance(self.N,int), "N must be an integer."
        assert (x0.shape[0]), "x0 is wrong size, must be 2x1."

    # change physical parameters
    def set_physical_params(self, g, z0):
        self.g = g
        self.z0 = z0
        lam = np.sqrt(g/z0)
        self.A = np.array([[0,1],[lam**2,0]])
    # change physical parameters
    def set_walking_params(self, z_apex, T):
        self.z_apex = z_apex   
        self.T = T             
    # change raibert controller parameters
    def set_raibert_params(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    # change trajectory settings
    def set_traj_settings(self, N, dt, x0):
        self.N = N     
        self.dt = dt   
        self.x0 = x0   

    # compute the LIP solution at time t given intial condition x0
    def lip_sol(self, t, x0):
        x_t = sp.linalg.expm(self.A*t) @ x0
        return x_t
    
    # reset map for LIP after completing swing phase
    def lip_reset(self, x,u):
        # reset map data
        p_stance = u
        p_CoM = x[0]
        v_CoM = x[1]

        # reassign state (i.e., apply reset map)
        p = p_CoM - p_stance
        v = v_CoM
        x = np.array([p,v])
        
        return x
    
    # compute foot placement target
    def foot_placement(self, p, v):
        u = self.alpha * v + self.beta*p  # super simple raibert heuristic
        return u
    
    # bezier curve for foot placement (for continuous updating)
    def foot_bezier(self,t,u_fp):
 
        # compute bezier curve control points, 5-pt bezier
        ctrl_pts_z = np.array([[0],[0],[(8/3)*self.z_apex],[0],[0]])
        ctrl_pts_x = np.array([[0],[0],[u_fp/2],[u_fp],[u_fp]])
        ctrl_pts = np.vstack((ctrl_pts_x.T,ctrl_pts_z.T))

        # evaluate bezier at time t (weird transofmration from drake)
        bezier = BezierCurve(0,self.T,ctrl_pts)
        b = np.array(bezier.value(t))       
        return np.array([b.T[0][0],b.T[0][1]])

    # make CoM and foot placement trajectories    
    def make_trajectories(self):

        # generate time series
        t_step = np.linspace(0,self.T,num=int(self.T/self.dt))

        # initialize trajectory containers
        t_list = []  # time list 
        x_list = []  # CoM state list
        b_list = []  # foot placement list

        # simulate the steps
        x0_current = self.x0
        t_current = 0.0

        # simulate LIP model
        for n in range(self.N):
            for t in t_step:
                
                # check if we are in stance or swing phase
                if (t > 0) and (t == self.T):
                    continue
                
                # get state and input at each time
                x = self.lip_sol(t,x0_current)
                u = self.foot_placement(x[0],x[1])
                b = self.foot_bezier(t,u)

                # append to lists
                t_list.append(t_current+t)
                x_list.append(x)
                b_list.append(b)

            # update time and state
            t_current += t
            x0_current = self.lip_reset(x,u)

        # convert to numpy arrays
        t_list = np.array(t_list).flatten()
        x_list = np.array(x_list)
        b_list = np.array(b_list)

        return t_list, x_list, b_list

############################# TESTING #############################

# test the code
N = 5
x0 = np.array([0.1,-0.1])
robot = LinearInvertedPendulum(N,x0)

# generate a trajecotry
[t,x,b] = robot.make_trajectories()

# Create a figure with three subplots
fig, axs = plt.subplots(3, 1, figsize=(6, 8))

# Plot the values of b in the first subplot
axs[0].plot(t, b)
axs[0].set_xlabel('Time')
axs[0].set_ylabel('b')
axs[0].grid(True)

# Plot the values of x[:,0] in the second subplot
axs[1].plot(t, x[:,0])
axs[1].set_xlabel('Time')
axs[1].set_ylabel('p_CoM - p_stance')
axs[1].grid(True)

# Plot the values of x[:,1] in the third subplot
axs[2].plot(t, x[:,1])
axs[2].set_xlabel('Time')
axs[2].set_ylabel('v_CoM')
axs[2].grid(True)

# Adjust the layout of the subplots
plt.tight_layout()

# Show the plot
plt.show()