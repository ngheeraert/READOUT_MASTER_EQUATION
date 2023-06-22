import numpy as np
from functions import *
from system_class import system
import matplotlib.pyplot as plt
import time 
#import qutip asqt
import sys
pi=np.pi

g = 0.05 * 2*pi
w = 5.5 * 2*pi
w01 = 5 * 2*pi
Ad= 0.037 * 2*np.pi * 0
wd = 5.5 * 2*np.pi
gamma=0.01 * 2*np.pi

tmax = 350
tint = 0.1
times = np.arange( 0, tmax, tint )
nsteps = len( times )

spinx_arr = np.zeros(  nsteps )
spinz_arr = np.zeros(  nsteps )
pop01 = np.zeros(  nsteps )
n_arr = np.zeros(  nsteps)

s = system( w01=w01,w=w, g=g, wd=wd,gamma=gamma,Ad=Ad,\
           cavity_dim=4, qubit_dim=2, dvice='TRSM1',\
           atol=1e-8,rtol=1e-6, max_step=1e-3, dim_exp=70, qb_ini=[1], coupling_type='00' )
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

