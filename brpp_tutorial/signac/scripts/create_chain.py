#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 12:39:56 2022

Create polydisperse stars. The script takes 4 arguments (f1,M1,f2,M2)
Provide the number of arms and the arm length as a pair (f,M) 

@author: utku
"""
import sys
import numpy as np


L = int(sys.argv[1])      # chain length

filename = 'start.data' 


if L % 2 != 0:
    M1 = (L-1) // 2
    M2 = (L-1) // 2
else:
    M1 = (L-1) // 2
    M2 = (L-1) // 2 + 1

f1 = 1
f2 = 1

num_atoms = f1*M1 + f2*M2 + 1
# chech if the number of atoms is correct
if (num_atoms) != L:
    print('The number of atoms is not equal to the chain length')
    sys.exit()
else:
    print('The number of atoms is equal to the chain length')

num_arms = f1 + f2

R = 0.5
D = 0.5

coor_core = np.zeros(3)

# Polymer parameters

blen1 = 0.97
dmin = 0.8
rho_polymer = 1.
monomer_mass = 1
core_mass = 1
N_polymer = f1*M1 + f2*M2

xlo = -10000 # box boundaries
xhi = 100000
ylo = -10000
yhi = 100000
zlo = -10000
zhi = 100000

coor_polymer1 = np.zeros((f1, M1, 3))

for j in range(f1):  # chain loop
    coor = np.zeros((M1,3)) 
    for i in range(M1):  # monomer loop for each chain
        if i==0:
            ph = np.pi * np.random.rand()
            th = 2 * np.pi * np.random.rand()
            xij = (R+0.5)*np.sin(ph)*np.cos(th) # initial position for each chain
            yij = (R+0.5)*np.sin(ph)*np.sin(th)
            zij = (R+0.5)*np.cos(ph)
        else:
            restriction = True  # coordinate of the first B atom for the first A atom
            while restriction:
                dx = 2.0 * np.random.rand() - 1.0
                dy = 2.0 * np.random.rand() - 1.0
                dz = 2.0 * np.random.rand() - 1.0
                rsq = dx**2 + dy**2 + dz**2
                r = np.sqrt(rsq)
                dx = dx / r
                dy = dy / r
                dz = dz / r
                xij = coor[i-1][0] + dx * blen1
                yij = coor[i-1][1] + dy * blen1
                zij = coor[i-1][2] + dz * blen1
                restriction = False
                # all monomers outside of NP
                if (np.sqrt(xij**2 + yij**2 + zij**2) < R+0.5):
                    restriction = True
                if i >= 2:
                    distx = xij - coor[i-2][0]
                    disty = yij - coor[i-2][1]
                    distz = zij - coor[i-2][2]
                    if (np.sqrt(distx*distx+ disty*disty + distz*distz) <= dmin): # the minimum distance between atom 1 and atom 3
                        restriction = True
                    elif ((xij<xlo) or (xij>xhi) or (yij<ylo) or (yij>yhi) or (zij<zlo) or (zij>zhi)):
                        restriction = True
                    #elif ((xij-5)**2 + (yij-5)**2 + (zij-5)**2) < 25 
                     #   restriction = True
        coor[i][0] = xij
        coor[i][1] = yij
        coor[i][2] = zij
    coor_polymer1[ j, : ] = coor[:,:]

coor_polymer1 = coor_polymer1.reshape(f1*M1, 3)
coor_core = coor_core.reshape((1,3))
coords = np.concatenate((coor_core, coor_polymer1))


coor_polymer2 = np.zeros((f2, M2, 3))

for j in range(f2):  # chain loop
    coor = np.zeros((M2,3)) 
    for i in range(M2):  # monomer loop for each chain
        if i==0:
            ph = np.pi * np.random.rand()
            th = 2 * np.pi * np.random.rand()
            xij = (R+0.5)*np.sin(ph)*np.cos(th) # initial position for each chain
            yij = (R+0.5)*np.sin(ph)*np.sin(th)
            zij = (R+0.5)*np.cos(ph)
        else:
            restriction = True  # coordinate of the first B atom for the first A atom
            while restriction:
                dx = 2.0 * np.random.rand() - 1.0
                dy = 2.0 * np.random.rand() - 1.0
                dz = 2.0 * np.random.rand() - 1.0
                rsq = dx**2 + dy**2 + dz**2
                r = np.sqrt(rsq)
                dx = dx / r
                dy = dy / r
                dz = dz / r
                xij = coor[i-1][0] + dx * blen1
                yij = coor[i-1][1] + dy * blen1
                zij = coor[i-1][2] + dz * blen1
                restriction = False
                # all monomers outside of NP
                if (np.sqrt(xij**2 + yij**2 + zij**2) < R+0.5):
                    restriction = True
                if i >= 2:
                    distx = xij - coor[i-2][0]
                    disty = yij - coor[i-2][1]
                    distz = zij - coor[i-2][2]
                    if (np.sqrt(distx*distx+ disty*disty + distz*distz) <= dmin): # the minimum distance between atom 1 and atom 3
                        restriction = True
                    elif ((xij<xlo) or (xij>xhi) or (yij<ylo) or (yij>yhi) or (zij<zlo) or (zij>zhi)):
                        restriction = True
                    #elif ((xij-5)**2 + (yij-5)**2 + (zij-5)**2) < 25 
                     #   restriction = True
        coor[i][0] = xij
        coor[i][1] = yij
        coor[i][2] = zij
    coor_polymer2[ j, : ] = coor[:,:]

coor_polymer2 = coor_polymer2.reshape(f2*M2, 3)
coords = np.concatenate((coords, coor_polymer2))


massA = core_mass
massB = 1


atomtypes = 1
bondtypes = 1

com = coords.mean(axis=0)

bxlo = com[0] - 50
bxhi = com[0] + 50
bylo = com[1] - 50
byhi = com[1] + 50
bzlo = com[2] - 50
bzhi = com[2] + 50

# Create molecule tag

molecule = np.ones(num_atoms)

# Charge

charge = np.zeros(num_atoms)

# Atom number                

num = np.arange(1,num_atoms+1)

# Type

types = np.ones(num_atoms)

coordinates = np.zeros((num_atoms,7))

coordinates[:, 0] = num
coordinates[:, 1] = molecule
coordinates[:, 2] = types
coordinates[:, 3] = charge
coordinates[:, 4:] = coords


### Bond matrix

# num_bonds_arm = arm_length - 1
# num_atoms = num_arms * arm_length + 1
# bonds = num_bonds_arm * num_arms + num_arms

num_bonds = num_atoms -1
bond = np.zeros((num_bonds, 4))
bond[:,0] = np.arange(1,num_bonds+1)

for k in range(f1):
    
    bond[ (k*(M1-1)):(k+1)*(M1-1), 1] = 1
    bond[ (k*(M1-1)):(k+1)*(M1-1), 2] = np.arange( k*M1+2, (k+1)*M1+1)
    bond[ (k*(M1-1)):(k+1)*(M1-1), 3] = np.arange( k*M1+3, (k+1)*M1+2)


for k in range(f2):
    
     bond[f1*(M1-1) + k*(M2-1) : f1*(M1-1) + (k+1)*(M2-1), 1 ] = 1
     bond[f1*(M1-1) + k*(M2-1) : f1*(M1-1) + (k+1)*(M2-1), 2 ] = np.arange( f1*M1+2 + k*M2, f1*M1+(k+1)*M2+1)
     bond[f1*(M1-1) + k*(M2-1) : f1*(M1-1) + (k+1)*(M2-1), 3 ] = np.arange( f1*M1+3 + k*M2, f1*M1+(k+1)*M2+2)    


# Bonds between arms and core

for k in range(f1):
    bond[ f1*(M1-1)+f2*(M2-1) + (k+1)-1, 1] = 1
    bond[ f1*(M1-1)+f2*(M2-1) + (k+1)-1, 2] = k * M1 + 2
    bond[ f1*(M1-1)+f2*(M2-1) + (k+1)-1, 3] = 1


for k in range(f2):
    bond[ f1*(M1-1)+f2*(M2-1)+f1 + (k+1)-1, 1] = 1
    bond[ f1*(M1-1)+f2*(M2-1)+f1 + (k+1)-1, 2] = (f1*M1) + k * M2 + 2
    bond[ f1*(M1-1)+f2*(M2-1)+f1 + (k+1)-1, 3] = 1


# Create LAMMPS data file                

with open(filename, 'w') as f:
    
    f.write(f'LAMMPS data file for coarse grained linear chain of length {num_atoms}  \n\n')
    f.write('%1.0f atoms\n' % num_atoms)
    f.write('%1.0f atom types\n' % atomtypes)
    f.write('%1.0f bonds\n' % num_bonds)            
    f.write('%1.0f bond types\n' % bondtypes)
    f.write('0 angles\n')
    f.write('0 angle types\n')
    f.write('0 dihedrals\n')            
    f.write('0 dihedral types\n')
    f.write('%1.6f %1.6f xlo xhi\n' % (bxlo, bxhi))
    f.write('%1.6f %1.6f ylo yhi\n' % (bylo, byhi))        
    f.write('%1.6f %1.6f zlo zhi\n\n' % (bzlo, bzhi))
    
    f.write('Masses\n\n')
    f.write('%1.0f %1.6f\n\n'% (1, massB))
    f.write('Atoms\n\n')
    np.savetxt(f, coordinates, fmt='%4.0f %6.0f %6.0f %6.2f %12.6f %12.6f %12.6f')
    f.write('\n')
    f.write('Bonds\n\n')
    np.savetxt(f, bond, fmt='%4.0f %6.0f %6.0f %6.0f')




















































