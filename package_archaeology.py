#!/usr/bin/env python

from subprocess import check_output
from pathlib import Path
import pandas as pd
import io

 

def parse_import(content):
	lines = content.split('\n')
	packages = []
	for line in lines:
		line = line.strip().replace('\"', '').replace('\\n', '').replace(',', ' ')
		if '#' in line or len(line.strip()) == 0:
			continue
		
		tokens = line.split()
		
		if 'from' in tokens:
			package = tokens[tokens.index('from') + 1]
		elif 'import' in tokens:
			package = tokens[tokens.index('import') + 1]
		else:
			print(f'Not likely an import line. Here is the tokens {tokens}. Decide for yourself.')
		package = package.split('.')[0]

		if len(package) > 0:
			packages.append(package)
		
	return packages


if __name__ == '__main__':
	pip_or_conda = 'conda'
	cmd = f'{pip_or_conda} list'
	columns = ['name', 'Version', 'Build', 'Channel']
	output = check_output(cmd, shell=True).decode('utf-8')
	data = io.StringIO(output)
	df = pd.read_csv(data, sep='\s+', comment='#', index_col=0, names=columns)
	print(df)


	fnames = list(Path('./').glob('**/*.py')) + list(Path('./').glob('**/*.ipynb'))	
	# fnames = ['cycleGan.ipynb', 'models/base_model.py', 'models/alexnet.py', 'models/resnet.py', 'utils/network.py']
	 
	packages_all = []
	for fname in fnames:
		self_stem = Path(__file__).stem
		if self_stem in str(fname):
			continue

		cmd = f"grep \"import\" {fname}"
		try:
			output = check_output(cmd, shell=True).decode('utf-8')
			packages = parse_import(output)	
			packages_all += packages
		except:
			continue
		
	requirements = []
	packages_all = set(packages_all)
	for i, package in enumerate(packages_all):
		try:
			version = df.loc[package, 'Version']
			print(f'{i + 1}: {package}, {version}')
			requirements.append(f'{package}>={version}')
		except:
			continue
	with open('requirements.txt', 'w') as fh:
		fh.write('\n'.join(requirements))
