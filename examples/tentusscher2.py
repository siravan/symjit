import os
import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np

from symjit import compile_json

path = os.path.join(os.path.dirname(__file__), "cellml", "tentusscher.json")
# path = os.path.join(os.path.dirname(__file__), "cellml", "beeler.json")

with open(path) as fd:
    model = fd.read()

f = compile_json(model, ty='bytecode')
g = compile_json(model)

u0 = f.get_u0()
p = f.get_p()

print(u0)

U = []

def F(t, u, *p):
    du = f(t, u, *p)
    U.append((t, u))
    return du

t_eval = np.arange(0, 2000, 1.0)

sol = scipy.integrate.solve_ivp(
    F, (0, 2000), u0, t_eval=t_eval, args=p, method="BDF", max_step=0.1
)

np.set_printoptions(precision=10)

for (t, u) in U:
    du_f = f(t, u, p)
    du_g = g(t, u, p)

    obs_f = f.compiler.obs
    obs_g = g.compiler.obs
    
    err = np.abs(du_f - du_g) 
    err2 = np.abs(obs_f - obs_g) 
    
    if max(err) > 1e-10 or max(err2) > 1e-10:
        print(t, ' fails')
        print('u:')
        print(u)
        
        
        print(err)
        print(err2)
        
        print('bytecode')
        print(du_f)
        
        print('compiled')
        print(du_g)
        
        break


