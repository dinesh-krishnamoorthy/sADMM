import numpy as np
from casadi import *
from numpy.linalg import norm
import math

opts = {}
opts["print_time"] = False
opts["ipopt"] = dict(print_level=1)

'''
Written: Dinesh Krishnamoorthy Dec 2020
'''

def MLP(u,w,nu,nn,ny):

    w1 = w[0:nu*nn]
    W1 = reshape(w1,nn,nu)
    b1 = w[nu*nn:nu*nn+nn]

    w2 = w[nu*nn+nn:nu*nn+nn+ny*nn]
    W2 = reshape(w2,ny,nn)
    b2 = w[nu*nn+nn+ny*nn:None]

    xii = mtimes(W1,transpose(u)) + b1
    xi = 1/(1 + exp(-xii))
    y = mtimes(W2,xi) + b2
    
    return y

def MLP_regression(u,y,nn,opts = opts):

    D,nu = u.shape
    D,ny = y.shape

    nw =  nu*nn+nn+ny*nn+ny
    
    x = MX.sym('x',nw)

    lbw = -math.inf*np.ones((nw,1))
    ubw = math.inf*np.ones((nw,1))
    w0 = np.random.rand(nw,1)

    lbg = []
    ubg = []

    L = 0

    for i in range(D):
        L  = L +  sumsqr(MLP(u[i,:].reshape(1,-1),x,nu,nn,ny) - y[i])
  
    L = L +  0.01*sumsqr(x)
    
    # Formulate NLP
    nlp = {'x':x, 'f':L,'g':[]}
    solver = nlpsol('solver', 'ipopt', nlp,opts)

    par = {'w0':w0,'lbw':lbw,'ubw':ubw,'lbg':lbg, 'ubg':ubg,'nlp':nlp}

    return solver,par

def solvenlp(solver,nlp,p=[]):
     # Get the optimal solution
    
    x0 = nlp['w0']
    lbw = nlp['lbw']
    ubw = nlp['ubw']
    lbg = nlp['lbg']
    ubg = nlp['ubg']
    sol = solver(x0 = x0, p = p, lbx=lbw, ubx=ubw, lbg = lbg, ubg = ubg)
    
    Primal = sol['x']
    Dual = {'lam_g':sol['lam_g'].full(),'lam_x':sol['lam_x'].full()} 
    f_opt = sol['f']
    
    flag = solver.stats()
    assert flag['success'] == 1,flag['return_status']
    return Primal, Dual, f_opt

def consensus_subproblem_nn_l1(u,y,nn,rho,opts = opts):
    
    D,nu = u.shape
    D,ny = y.shape
    
    nw =  nu*nn+nn+ny*nn+ny

    x = MX.sym('x',nw)
    lbw = -math.inf*np.ones(nw)
    ubw = math.inf*np.ones(nw)
    w0 = 0*np.ones(nw)

    lam = MX.sym('lam',nw)
    x0 = MX.sym('x0',nw)
    
    # Constraint bounds
    lbg = []
    ubg = []

    L = 0

    for i in range(D):
        L  = L + sumsqr(MLP(u[i,:].reshape(1,-1),x,nu,nn,ny) - y[i])

    L = L + rho/2*sumsqr(x-x0+lam)

    # Formulate QP
    nlp = {'x':x, 'p':vertcat(lam,x0),'f':L ,'g':[]}
    solver = nlpsol('solver', 'ipopt', nlp,opts)

    par = {'w0':w0,'lbw':lbw,'ubw':ubw,'lbg':lbg, 'ubg':ubg,'nlp':nlp}
    return solver,par

def soft_threshold(a,kappa):
    return np.maximum(0,a-kappa) - np.maximum(0,-a-kappa)

def ADMM_consensus_l1(solvers,nlps,x0_opt,Lam,rho,tol=1e-3,MaxIter=50):
    
    assert len(solvers) == len(nlps)
    # assert x0_opt is the same size as nlps[i].w0
    dLambda = 100 # inital large dLambda
    Lam0 = vertcat(*Lam)  
    k = 0
    
    r_primal = []
    r_dual = []
    while dLambda>tol and k<=MaxIter:
        k = k+1
        x_opt = []
        
        # Solve Direction 1
        for i in range(0,len(solvers)):
            xi_opt,_,_ = solvenlp(solvers[i],nlps[i],vertcat(Lam[i],x0_opt))
            x_opt.append(xi_opt)

        x0_opt0 = x0_opt
        # solve Direction 2
        a = 0
        for i in range(0,len(solvers)):
            a += x_opt[i] + Lam[i]
 
        x0_opt = soft_threshold(a.full()/4,1/(4*rho))
        
        # Update Lagrange multipliers
        for i in range(0,len(solvers)):
            Lam[i] += (x_opt[i] - x0_opt)

        dLambda = norm(Lam0  - vertcat(*Lam))
        Lam0 = vertcat(*Lam)

        res = 0
        for i in range(0,len(solvers)):
            res += np.linalg.norm(x_opt[i]-x0_opt)
        r_primal.append(res)
        r_dual.append(np.linalg.norm(rho*(x0_opt-x0_opt0)))
    return x0_opt,x_opt,dLambda,r_primal,r_dual

def sADMM_consensus_l1(solvers,nlps,x0_opt,Lam,rho,tol=1e-3,MaxIter=50,):
    
    assert len(solvers) == len(nlps)
    # assert x0_opt is the same size as nlps[i].w0
    dLambda = 100 # inital large dLambda
    Lam0 = vertcat(*Lam)  
    k = 0

    r_primal = [20]
    r_dual = [20]
    r_app = [200]

    while k<=MaxIter:
        k = k+1
        
        # Solve Direction 1
        if  r_dual[k-1]>500 or r_primal[k-1]>10:
            x_opt = []
            dual = []
            eps=[]
            for i in range(0,len(solvers)):
                xi_opt,duali_opt,_ = solvenlp(solvers[i],nlps[i],vertcat(Lam[i],x0_opt))
                x_opt.append(xi_opt)
                dual.append(duali_opt)
                eps.append(0)
            
        else:
                
            for i in range(0,len(solvers)):
                dPrimal, duali_opt,Lx = predictor_tangent(x_opt[i],dual[i],p_init[i],p_final[i],nlps[i])
                xi_opt = x_opt[i] + dPrimal
                x_opt[i] = xi_opt
                dual[i] = duali_opt    
                eps[i] = np.linalg.norm(Lx(x_opt[i],p_final[i]))
        
        p_init = []
        for i in range(0,len(solvers)):
            p_init.append(vertcat(Lam[i],x0_opt))

        x0_opt0 = x0_opt
        # solve Direction 2
        a = 0*x_opt[1]
        for i in range(0,len(solvers)):
            a += x_opt[i] + Lam[i]
 
        x0_opt = soft_threshold(a.full()/len(solvers),1/(len(solvers)*rho))

        # Update Lagrange multipliers
        for i in range(0,len(solvers)):
            Lam[i] += (x_opt[i] - x0_opt)

        p_final = []
        for i in range(0,len(solvers)):
            p_final.append(vertcat(Lam[i],x0_opt))

        dLambda = norm(Lam0  - vertcat(*Lam))
        Lam0 = vertcat(*Lam)

        res = 0
        for i in range(0,len(solvers)):
            res += np.linalg.norm(x_opt[i]-x0_opt)
        r_primal.append(res)
        r_dual.append(4*np.linalg.norm(rho*(x0_opt-x0_opt0)))
        r_app.append(np.linalg.norm(eps))

    return x0_opt,x_opt,r_primal,r_dual,r_app

def get_sensitivities(par):
    nlp = par['nlp']
    w = nlp['x']
    p = nlp['p']
    J = nlp['f']

    Lagr_func = J  #+ sum1(Dual['lam_x']*w)

    Lpx = Function('Lpx',[w,p],[jacobian(jacobian(Lagr_func,p),w)],['w','p'],['Lpw'])
    H = Function('H',[w,p],[jacobian(jacobian(Lagr_func,w),w)],['w','p'],['Lww'])
    Lx = Function('Lx',[w,p],[jacobian(Lagr_func,w)],['w','p'],['Lpw'])
    
    return H,Lpx,Lx

def predictor_tangent(Primal,Dual,p_init,p_final,par):
    
    nw = Primal.shape[0]
    ng = Dual['lam_g'].shape[0]

    dp = (p_final-p_init)
    H,Lpx,Lx = get_sensitivities(par)
    
    H = H(Primal,p_init)
    Lpx = Lpx(Primal,p_init)

    M = H.full()
    N = np.matmul(np.transpose(Lpx.full()),dp.full())

    Delta_s = np.linalg.solve(M,-1*N)
    
    dPrimal = Delta_s[0:nw]
    Dual['lam_g'] = Delta_s[nw:nw+ng]
    Dual['lam_x'] = Delta_s[nw+ng:nw+ng+nw]
    
    return dPrimal, Dual, Lx

def stochastic_sADMM_consensus_l1(solvers,nlps,solver0,nlp0,x0_opt,Lam,rho,tol=1e-3,MaxIter=50,delta = 0.5):
    
    assert len(solvers) == len(nlps)
    # assert x0_opt is the same size as nlps[i].w0
    dLambda = 100 # inital large dLambda
    Lam0 = vertcat(*Lam)  
    k = 0
    
    r_primal = [20]
    r_dual = [20]
    r_app = [200]
    while k<=MaxIter:
        k = k+1
        
        p = np.random.uniform(0,1)
        if p<=delta**(k-1):
            nlp = True
        else:
            nlp = False

        # Solve Direction 1
        if  r_dual[k-1]>500 or r_primal[k-1]>10 or nlp:
            x_opt = []
            dual = []
            eps=[]
            for i in range(0,len(solvers)):
                xi_opt,duali_opt,_ = solvenlp(solvers[i],nlps[i],vertcat(Lam[i],x0_opt))
                x_opt.append(xi_opt)
                dual.append(duali_opt)
                eps.append(0)
            
        else:
            for i in range(0,len(solvers)):
                dPrimal, duali_opt,Lx = predictor_tangent(x_opt[i],dual[i],p_init[i],p_final[i],nlps[i])
                xi_opt = x_opt[i] + dPrimal
                x_opt[i] = xi_opt
                dual[i] = duali_opt    
                eps[i] = np.linalg.norm(Lx(x_opt[i],p_final[i]))
        
        p_init = []
        for i in range(0,len(solvers)):
            p_init.append(vertcat(Lam[i],x0_opt))

        x0_opt0 = x0_opt
        # solve Direction 2
        a = 0*x_opt[1]
        for i in range(0,len(solvers)):
            a += x_opt[i] + Lam[i]
 
        x0_opt = soft_threshold(a.full()/len(solvers),1/(len(solvers)*rho))

        # Update Lagrange multipliers
        for i in range(0,len(solvers)):
            Lam[i] += (x_opt[i] - x0_opt)

        p_final = []
        for i in range(0,len(solvers)):
            p_final.append(vertcat(Lam[i],x0_opt))

        dLambda = norm(Lam0  - vertcat(*Lam))
        Lam0 = vertcat(*Lam)

        res = 0
        for i in range(0,len(solvers)):
            res += np.linalg.norm(x_opt[i]-x0_opt)
        r_primal.append(res)
        r_dual.append(4*np.linalg.norm(rho*(x0_opt-x0_opt0)))
        r_app.append(np.linalg.norm(eps))
    return x0_opt,x_opt,r_primal,r_dual