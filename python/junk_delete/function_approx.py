import numpy as np
from casadi import *
from numpy.linalg import norm
import math


opts = {}
opts["print_time"] = False
opts["ipopt"] = dict(print_level=0)

def ridge(A,b,opts = opts):
    
    m,n = A.shape
    
    lbw = -1e10*np.ones(n)
    ubw = 1e10*np.ones(n)
    w0 = 0*np.ones(n)

    x = MX.sym('x',n)
    
    # Constraint bounds
    lbg = []
    ubg = []

    A = A.to_numpy()
    b = b.to_numpy()
    
    X =[]
    for i in range(0,n):
        X.append(x[i])
    
    f =  vertcat(*np.matmul(A,X))
    L =  sumsqr((f - b)) + sumsqr(x)
    
    # Formulate QP
    qp = {'x':x, 'f':L,'g':[]}
    solver = nlpsol('solver', 'ipopt', qp,opts)

    nlp = {'w0':w0,'lbw':lbw,'ubw':ubw,'lbg':lbg, 'ubg':ubg,'nlp':qp}
    return solver,nlp


