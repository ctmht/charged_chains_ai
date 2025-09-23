from matplotlib import pyplot as plt
import pandas as pd

import os


if __name__ == '__main__':
	current_folder = os.path.dirname(os.path.abspath(__file__))
	processed_file = os.path.join(current_folder, 'simulation_prototype/9_processed.pkl')
	
	df = pd.read_pickle(processed_file)
	
	plt.figure()
	
	for row in df['end_to_end_autocorrelations']:
		plt.plot(row)
		plt.hist(row, bins=25, orientation='horizontal', alpha=0.75, zorder=2)
	
	plt.show()