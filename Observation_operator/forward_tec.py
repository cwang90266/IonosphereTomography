import numpy as np

def abel_ne_state(abel_ne_m3,n_geo):
    return np.tile(np.asarray(abel_ne_m3,float)[:,None],(1,int(n_geo))).reshape(-1)

def forward_tec(H,state):
    return np.asarray(H,float)@np.asarray(state,float)
