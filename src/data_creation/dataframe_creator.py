from typing import Callable, Literal
import os

import pandas as pd
import numpy as np


def create_df(
	infile: str,
	outfile: str,
	shuffle: bool = False
) -> None:
	"""
	
	"""
	with open(infile, "r") as seq_f:
		seqs = seq_f.readlines()
		seqs = pd.Series(seqs).apply(lambda entry: entry[:-1])
		
		df = pd.DataFrame({
			"sequence": seqs
		})
		
		def get_num(monomer_type: str) -> Callable:
			def internal_get_num(seq: str) -> int:
				count = 0
				for character in seq:
					count += (character == monomer_type)
				return count
			return internal_get_num
		
		df["N_A"] = df["sequence"].apply(get_num("A")) # get number of aliphatic monomers
		df["N_C"] = df["sequence"].apply(get_num("C")) # get number of anion monomers
		
		if shuffle:
			df = df.sample(frac=1).reset_index(drop=True)
		
		df.to_csv(outfile)
		df.to_pickle(outfile)



if __name__ == '__main__':
	DATA_FOLDER = os.path.abspath("./data/")
	
	# targets = ["autocorr", "full"]
	targets = ["full"]
	
	SHUFFLE = True
	
	for taskname in targets:
		seqpath = os.path.join(DATA_FOLDER, f"sequences_{taskname}_revsym.txt")
		dfpath = os.path.join(DATA_FOLDER, f"{taskname}_dataframe.pkl")
		create_df(seqpath, dfpath, SHUFFLE)