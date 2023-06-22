import numpy as np
from functions import *
from system_class import system
import matplotlib.pyplot as plt
import time 
#import qutip asqt
import sys
pi=np.pi

g=  0.0486 * 2*pi
w = 7.415 * 2*pi
#w = 5.401976741795615*2*pi
Ad=0.5 * 2*np.pi
#wd_qb0 = (7.41560) * 2*np.pi
#wd_qb1= (7.41439) * 2*np.pi
wd = (4.4619) * 2*np.pi
gamma=0.0014 * 2*np.pi

tmax = 300
tint = 0.1
times = np.arange( 0, tmax, tint )
nsteps = len( times )

spinx_arr = np.zeros(  nsteps )
spinz_arr = np.zeros(  nsteps )
pop01 = np.zeros(  nsteps )
n_arr = np.zeros(  nsteps)

s = system( g=g,w=w,wd=wd,gamma=gamma,Ad=Ad,\
           cavity_dim=5, qubit_dim=4, dvice='TRSM1',\
           atol=1e-8,rtol=1e-6, max_step=1e-3, dim_exp=70, qb_ini=[0], coupling_type='00' )
s.set_initial_cs_state( 0 )
s.initialise_density_matrix()


print('-- paramachar:')
print(s.paramchar(times[-1]))
print('====================================')

t1 = time.time()
rhos = s.time_evolve( times )
t2 = time.time()

print('-- total time:', t2-t1)

s.save_and_plot( times, rhos, [0], max_lambda=7 )
#s.save_husimi_for_gif( times, rhos, max_lambda=7, frame_nb=100, log=True )

fidelity = s.calc_fidelity( rhos[-1] )
print( 'qubit fidelity =', fidelity )
