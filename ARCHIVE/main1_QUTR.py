import numpy as np
from functions import *
from system_class import system
import matplotlib.pyplot as plt
import time 
#import qutip asqt
import sys
pi=np.pi

g=  0.0012 * 2*pi
w = 7.415 * 2*pi
#Ad=0.006 * 2*np.pi
Ad=0.0059 * 2*np.pi
wd=7.413199 * 2*np.pi
#wd=7.4138 * 2*np.pi
gamma=0.0014 * 2*np.pi

tmax = 1000
tint = 0.1
times = np.arange( 0, tmax, tint )
nsteps = len( times )

spinx_arr = np.zeros(  nsteps )
spinz_arr = np.zeros(  nsteps )
pop01 = np.zeros(  nsteps )
n_arr = np.zeros(  nsteps )

s = system( g=g,w=w,wd=wd,gamma=gamma,Ad=Ad,\
           cavity_dim=30, qubit_dim=4, dvice='QUTR1',\
           atol=1e-8,rtol=1e-6, max_step=1e-3, dim_exp=50, qb_ini=[1], coupling_type='00' )
s.set_initial_cs_state( 0 )
s.initialise_density_matrix()


print('-- paramachar:')
print(s.paramchar(times[-1]))
print('====================================')

t1 = time.time()
rhos = s.time_evolve( times )
t2 = time.time()

print('-- total time:', t2-t1)

s.save_and_plot( times, rhos, [0], max_lambda=5.5 )
#s.save_husimi_for_gif( times, rhos, max_lambda=7, frame_nb=2, log=True )

