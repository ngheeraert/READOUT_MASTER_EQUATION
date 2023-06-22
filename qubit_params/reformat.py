import numpy as np
import copy

energy_filename = "Elvl_Ec-0.221_wq5.4.txt"
charge_op_filename = "charge_op_Ec-0.221_wq5.4.txt"

charge_op = np.loadtxt( charge_op_filename )
energy = np.loadtxt( energy_filename )

np.savetxt( 'FOR_'+energy_filename, energy, fmt='%20.15f',delimiter='' )
np.savetxt( 'FOR_'+charge_op_filename, charge_op, fmt='%20.15f',delimiter='' )




