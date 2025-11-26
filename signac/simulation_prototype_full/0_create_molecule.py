from typing import Literal
import sys

import numpy as np


def create_molecule(
	sequence: str,
	charge_map: dict[str, float],
	filetype: Literal['mol', 'data'] = 'data'
) -> None:
	"""
	Create a molecule representing the given sequence of atoms with predefined
	characteristics
	
	:param sequence: the sequence of atom types to use
	:param charge_map: charge of each atom type found in the sequence
	
	:return: None
	"""
	
	# All atoms must have the charge accounted for
	unique_types = set(sequence)
	diff = unique_types - set(charge_map.keys())
	if len(diff) > 0:
		raise ValueError(f'Undefined charges for types {diff}')
	
	unique_types = sorted(unique_types)
	
	# Numpy array for easier management
	sequence_np = np.array(list(sequence))
	
	
	chain_length: int = len(sequence)
	bond_length: float = 0.97 # TODO: bond length?
	
	# Get initial coordinates for the atoms
	atom_coords = np.arange(start=0, stop=chain_length*bond_length, step=bond_length)
	
	# Get ordered list of atom types as integers by mapping chars to ints
	atom_types = np.zeros_like(sequence_np, dtype=int)
	for atom_strtype in unique_types:
		atom_types[sequence_np==atom_strtype] = unique_types.index(atom_strtype) + 1
	atom_types = atom_types.astype(int)
	
	# Get charges per atom type
	atom_charges = np.zeros_like(sequence_np, dtype=float)
	for atom_strtype in unique_types:
		atom_charges[sequence_np==atom_strtype] = charge_map[atom_strtype]
	atom_charges = atom_charges.astype(float)
	
	# Total charge must be zero
	if (tot_charge := np.sum(atom_charges)) != 0:
		raise ValueError(f'Total chain charge is {tot_charge} instead of 0')
	
	# Get bonds, defined between consecutive atoms (only one bond type)
	bond_at1 = np.arange(1, chain_length, 1, dtype=int)
	
	# print(atom_types, atom_charges, bond_at1, sep='\n')
	
	
	filename = f'1_polymer.{filetype}'
	with open(filename, 'w') as f:
		if filetype == 'mol':
			f.write(f'# Polymer sequence:\n# {sequence}\n')
			f.write(f'{chain_length} atoms\n')
			f.write(f'{chain_length - 1} bonds\n')
			
			f.write('\n# Atom types: [atomID] [typeID]')
			f.write('\nTypes\n\n')
			for idx, type in enumerate(atom_types):
				f.write(f'{idx + 1 :<5} {type}\n')
			
			f.write('\n# Coordinates: [atomID] [x] [y] [z]')
			f.write('\nCoords\n\n')
			for idx, pos in enumerate(atom_coords):
				f.write(f'{idx + 1 :<5} {pos :<7.5f} {0} {0}\n')
						
			f.write('\n# Charges: [atomID] [charge]')
			f.write('\nCharges\n\n')
			for idx, charge in enumerate(atom_charges):
				f.write(f'{idx + 1 :<5} {charge :>5.1f}\n')
			
			f.write('\n# Bonds: [bondID] [bondtypeID] [atom1ID] [atom2ID]')
			f.write('\nBonds\n\n')
			for idx, (at1, at2) in enumerate(zip(bond_at1, bond_at1 + 1)):
				f.write(f'{idx + 1 :<5} {1 :<5} {at1 :<5} {at2 :<5}\n')
		
		elif filetype == 'data':
			f.write('LAMMPS Description\n\n')
			f.write(f'# Polymer sequence:\n# {sequence}\n')
			
			f.write(f'{chain_length} atoms\n')
			f.write(f'{chain_length - 1} bonds\n')
			f.write(f'{len(set(atom_types))} atom types\n')
			f.write(f'1 bond types\n\n')
			# ^ 1 bond type, but interactions differ: will be done by pair_style
			
			# Box size (approximate)
			xhi = atom_coords.max() + 10
			yhi = atom_coords.max() + 10
			f.write(f'0.0 {xhi :.2f} xlo xhi\n')
			f.write(f'0.0 {yhi :.2f} ylo yhi\n')
			f.write(f'-5.0 5.0 zlo zhi\n\n')
			
			# [typeID] [mass]
			f.write('Masses\n\n')
			for atom_type in sorted(set(atom_types)):
				f.write(f'{atom_type} 1.0\n')
			f.write('\n')
			
			# [atom_id] [molecule_tag] [atom_type] [charge] [pos_x] [pos_y] [pos_z]
			f.write('Atoms # full\n\n')
			for idx in range(chain_length):
				f.write(f'{idx + 1 :<5} 1 {atom_types[idx]} {atom_charges[idx] :>5.1f} '
						f'{atom_coords[idx] :<7.5f} 0.0 0.0\n')
			
			f.write('\nBonds\n\n')
			for idx, (at1, at2) in enumerate(zip(bond_at1, bond_at1 + 1)):
				f.write(f'{idx + 1 :<5} {1 :<5} {at1 :<5} {at2 :<5}\n')
	
	f.close()
	print('File written:', filename)
		


# bond_style fene
# bond_coeff * [K] [R_0] [eps] [sig]
# K (energy/distance^2): attractive factor
# R_0 (distance): attractive cutoff distance
# eps (energy): LJ potential (repulsive) energy
# sig (distance): LJ cutoff (LJ cutoff at 2^(1/6) sig)


charge_map_global = {
	'A': 0,		# atom type 1: aliphatic
	'B': 0,		# atom type 2: aromatic
	'C': -1,	# atom type 3: anion
	'D': +1		# atom type 4: cation
}


if __name__ == '__main__':
	seq = sys.argv[1]
	print(seq)
	create_molecule(seq, charge_map_global, 'mol')
	create_molecule(seq, charge_map_global, 'data')