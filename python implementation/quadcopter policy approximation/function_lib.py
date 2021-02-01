import numpy as np
from casadi import *

'''
Library of useful CasADi functions written and collected over time

- solveODE(sys,u_in,dist,w0,lbw,ubw,opts=opts)
- SS_opt(sys,dist,w0,opts=opts )
- DirectCollocation(d=3)
- buildNLP(sys,par,N=60,tf=1,d=3,opts=opts)

Written by Dinesh Krishnamoorthy
'''

opts = {}
opts["print_time"] = False
opts["ipopt"] = dict(print_level=1)

def quadcopter(Ts,m = 1.846, l = 0.505, beta = 0.2):
    '''
    - Nonlinear Quadcopter dynamics
    - Bemporad, A., Pascucci, C.A. and Rocchi, C., 2009. Hierarchical and hybrid model predictive control of quadcopter air vehicles. IFAC Proceedings Volumes, 42(17), pp.14-19.
    - Written by D. Krishnamoorthy Jan 2021
    '''

    x = MX.sym('x',12)  # 12 states
    u = MX.sym('u',4) # 4 inputs - thrust forces generated by the four motors
    
    xsp = MX.sym('xsp')
    ysp = MX.sym('ysp')
    zsp = MX.sym('zsp')
    d = vertcat(xsp,ysp,zsp) # target coordinates

    Ixx = Iyy = 0.1722 # elements of the diagonal Inertia matrix
    Izz = 0.3424 
    F = u[0] #+u[1]+u[2]+u[3] # Total thrust force

    dx1 = x[6]  # angular velocity \dot\theta
    dx2 = x[7]  # angular velocity \dot\phi
    dx3 = x[8]  # angular velocity \dot\psi
    dx4 = x[9]  # velocity \dot\x
    dx5 = x[10] # velocity \dot\y
    dx6 = x[11] # velocity \dot\z
    dx7 = u[1]#(u[1]-u[3])*l/Ixx  # torque \tau_\theta
    dx8 = u[2]#(u[2]-u[0])*l/Iyy  # torque \tau_\phi
    dx9 = u[3]#(-u[0]+u[1]-u[2]+u[3])*l/Izz  # torque \tau_\psi
    dx10 = (-F*sin(x[0]) - beta*x[9])/m
    dx11 = (F*cos(x[0])*sin(x[1]) - beta*x[10])/m
    dx12 = -9.81 + (F*cos(x[0])*cos(x[1]) - beta*x[11])/m

    dx = vertcat(dx1,dx2,dx3,dx4,dx5,dx6,dx7,dx8,dx9,dx10,dx11,dx12)
    
    L =  10*(x[3] - xsp)**2 + 10*(x[4] - ysp)**2 + 10*(x[5] - zsp)**2 + sumsqr(x[0:3]) + sumsqr(x[6:12])
        
    # create CVODES integrator
    ode = {'x':x,'p':vertcat(u,d),'ode':dx,'quad':L}
    opts = {'tf':Ts}
    F = integrator('F','cvodes',ode,opts) 

    f = Function('f',[x,vertcat(u,d)],[dx,L],['x','p'],['xdot','qj'])
        
    sys = {'x':x,'u':u,'d':d,'dx':dx,'L':L,'f':f}
    return sys,F


def solveODE(sys,u_in,dist,opts=opts):
    
    assert dist.shape[0] == sys['d'].shape[0], "Dimension mismatch."
    assert u_in.shape[0] == sys['u'].shape[0], "Dimension mismatch."

    w0 = np.zeros(sys['x'].shape).reshape(-1,)
    lbw = -inf*np.ones(sys['x'].shape).reshape(-1,)
    ubw = inf*np.ones(sys['x'].shape).reshape(-1,)

    # Constraints
    g = []
    lbg = []
    ubg = []
    
    g.append(sys['dx'])
    lbg.append(np.zeros(sys['x'].shape).reshape(-1,))
    ubg.append(np.zeros(sys['x'].shape).reshape(-1,))

    lbg = np.concatenate(lbg)
    ubg = np.concatenate(ubg)

    nlp = {'x':sys['x'], 'p':vertcat(sys['u'],sys['d']), 'f':0, 'g':vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', nlp,opts)
    sol = solver(x0=w0,p=vertcat(u_in,dist), lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    xf = sol['x'].full()
    return xf

def SS_opt(sys,dist,opts=opts ):
    
    assert dist.shape[0] == sys['d'].shape[0], "Dimension mismatch."
    
    dx0 = np.zeros(sys['x'].shape).reshape(-1,)
    lbx = -inf*np.ones(sys['x'].shape)
    ubx = inf*np.ones(sys['x'].shape)

    u0 = np.zeros(sys['u'].shape).reshape(-1,)
    lbu = -inf*np.ones(sys['u'].shape)
    ubu = inf*np.ones(sys['u'].shape)

    # Variables
    w = []
    lbw = []
    ubw = []

    # Constraints
    g = []
    lbg = []
    ubg = []

    w.append(sys['x'])
    w.append(sys['u'])
    lbw.append(np.concatenate([lbx.reshape(-1,),lbu.reshape(-1,)]).tolist())
    ubw.append(np.concatenate([ubx.reshape(-1,),ubu.reshape(-1,)]).tolist())

    g.append(sys['dx'])
    lbg.append(np.zeros(sys['x'].shape).reshape(-1,))
    ubg.append(np.zeros(sys['x'].shape).reshape(-1,))

    lbw = np.concatenate(lbw)
    ubw = np.concatenate(ubw)
    lbg = np.concatenate(lbg)
    ubg = np.concatenate(ubg)

    nlp = {'x':vertcat(*w), 'p':vertcat(sys['d']), 'f':sys['L'], 'g':vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', nlp,opts)
    sol = solver(x0=vertcat(dx0,u0),p=dist, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    
    return sol,solver.stats()['return_status']

def DirectCollocation(d=3):
    # Get collocation points
    tau_root = np.append(0, collocation_points(d, 'legendre'))

    # Coefficients of the collocation equation
    C = np.zeros((d+1,d+1))

    # Coefficients of the continuity equation
    D = np.zeros(d+1)

    # Coefficients of the quadrature function
    B = np.zeros(d+1)

    # Construct polynomial basis
    for j in range(d+1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d+1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)

        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(d+1):
            C[j,r] = pder(tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)
        
    return B,C,D

def buildNLP(sys,par,N=60,tf=1,d=3,MVROC= 0,opts=opts):
    
    f = sys['f']
    
    nx = sys['x'].shape[0]
    nu = sys['u'].shape[0]
    nd = sys['d'].shape[0]
    
    lbx = par['lbx']
    ubx = par['ubx']
    dx0 = par['dx0']
    
    lbu = par['lbu']
    ubu = par['ubu']
    u0 = par['u0']
    
    B,C,D = DirectCollocation(d)
    # Start with an empty NLP
    w=[]
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g=[]
    lbg = []
    ubg = []

    # For plotting x and u given w
    x_plot = []
    u_plot = []
    
    # Intial condition 

    Xk = MX.sym('Xk', nx)
    w.append(Xk)
    lbw.append(lbx)
    ubw.append(ubx)
    w0.append(dx0)
    x_plot.append(Xk)
    
    
    xf  = MX.sym('xf',nx)
    U0  = MX.sym('U0',nu)
    Dk    = MX.sym('Dk',nd)
    
    Uk_prev = U0
    
    # Initial condition constraint
    g.append(Xk-xf)
    lbg.append(np.zeros(sys['x'].shape).reshape(-1,))
    ubg.append(np.zeros(sys['x'].shape).reshape(-1,))
    
    for k in range(N):
        Uk = MX.sym('U_' + str(k),nu)
        w.append(Uk)
        lbw.append(lbu)
        ubw.append(ubu)
        w0.append(u0)
        u_plot.append(Uk)
        
         # State at collocation points
        Xc = []
        for j in range(d):
            Xkj = MX.sym('X_'+str(k)+'_'+str(j), nx)
            Xc.append(Xkj)
            w.append(Xkj)
            lbw.append(lbx)
            ubw.append(ubx)
            w0.append(dx0)

        # Loop over collocation points
        Xk_end = D[0]*Xk
        
        for j in range(1,d+1):
            # Expression for the state derivative at the collocation point
            xp = C[0,j]*Xk
            for r in range(d): 
                xp = xp + C[r+1,j]*Xc[r]

            # Append collocation equations
            fj, qj = f(Xc[j-1],vertcat(Uk,Dk))
            g.append(tf*fj - xp)
            lbg.append(np.zeros(sys['x'].shape).reshape(-1,))
            ubg.append(np.zeros(sys['x'].shape).reshape(-1,))

            # Add contribution to the end state
            Xk_end = Xk_end + D[j]*Xc[j-1]

            # Add contribution to quadrature function
            J = J + B[j]*qj*tf
            
        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str(k+1), nx)
        w.append(Xk)
        lbw.append(lbx)
        ubw.append(ubx)
        w0.append(dx0)
        x_plot.append(Xk)

        # Shooting gap constraint
        g.append(Xk_end-Xk)
        lbg.append(np.zeros(sys['x'].shape).reshape(-1,))
        ubg.append(np.zeros(sys['x'].shape).reshape(-1,))
        
    # Concatenate vectors
    w = vertcat(*w)
    g = vertcat(*g)
    x_plot = horzcat(*x_plot)
    u_plot = horzcat(*u_plot)
    w0 = np.concatenate(w0)
    lbw = np.concatenate(lbw)
    ubw = np.concatenate(ubw)
    lbg = np.concatenate(lbg)
    ubg = np.concatenate(ubg)

    # Create an NLP solver
    nlp = {'x': w, 'p': vertcat(xf,U0,Dk), 'g': g,'f': J}
    solver = nlpsol('solver', 'ipopt', nlp,opts)

    # Function to get x and u trajectories from w
    trajectories = Function('trajectories', [w], [x_plot, u_plot], ['w'], ['x', 'u'])
    
    OCP = {'solver':solver,'nlp':nlp,'x0':w0,'lbx':lbw,'ubx':ubw,'lbg':lbg,'ubg':ubg}
    return OCP, trajectories