import numpy as np
from functions import *
from system_class import system
import matplotlib.pyplot as plt
import time 
#import qutip asqt
import sys
import os
pi=np.pi

#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['MKL_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1'

#g = 0.1586 * 2*pi
g = 0.05 * 2*pi
w = 7.415 * 2*pi
#w = 5.401976741795615*2*pi
Ad = 0.006 * 2*np.pi
wd = 7.415 * 2*np.pi
gamma = 0.0014 * 2*np.pi
#Omega=0.002 * 2*np.pi

tmax = 1200
tint = 1
times = np.arange( 0, tmax, tint )
nsteps = len( times )

spinx_arr = np.zeros(  nsteps )
spinz_arr = np.zeros(  nsteps )
pop01 = np.zeros(  nsteps )
n_arr = np.zeros(  nsteps)

s = system( g=g,w=w,wd=wd,gamma=gamma,Ad=Ad,\
           cavity_dim=30, qubit_dim=2, dvice='TRSM1',\
           atol=1e-8,rtol=1e-6, max_step=1e-3, dim_exp=50, qb_ini=[0,1], coupling_type='00' )
s.set_initial_cs_state( alpha=0 ) 
s.initialise_density_matrix()


print('-- paramachar:')
print(s.paramchar(times[-1]))
print('====================================')

t1 = time.time()
rhos = s.time_evolve( times )
t2 = time.time()

print('-- total time:', t2-t1)

s.save_and_plot( times, rhos, [0], max_lambda=5 )
s.save_husimi_for_gif( times, rhos, max_lambda=5, frame_nb=100, log=False )
