import subprocess
import os
import errno
import json
import numpy as np
import time
import datetime

# Seed the random generator
np.random.seed(int(round(time.time())))


def run2(no_voxels=1000):
	cylinder_rads = [1e-12, 1e-11, 5e-10, 2e-9, 5e-9]
	print 'Begin data generation with ' + str(len(cylinder_rads)) + ' iterations and ' + str(no_voxels) + ' in every iteration'
	for i in range(0, len(cylinder_rads)):
		print('Running iteration ' + str(i+1) + ' of ' + str(len(cylinder_rads)))
		print(time.strftime("%c"))
		dirname = "./data/search/gen" + str(i) + '-' + str(datetime.datetime.now().isoformat()) + '/'

		# Get radius
		radius = cylinder_rads[i]

		# Get a config file from describing the generated data
		config = _get_config(voxels=no_voxels, cylinder_rad=radius, cylinder_sep=1.1E-6, dir_name=dirname)
		# Perform the actual data generation
		_generate_data(config)


def run(no_iter=100, no_voxels=1000, cylinder_rad_from=1E-8, cylinder_rad_to=1E-6, cylinder_sep_from=1.1E-6, cylinder_sep_to=1.1E-6):
	"""
	Main method for generating data with camino
	Calls bash-script which in turn calls camino to generate data 
	@param no_iter: Number of 
	@param no_voxels: Number of voxels to generate
	@param cylinder_rad_from: random cylinder radius in range from
	@param cylinder_rad_to: random cylinder radius in range to
	@param cylinder_sep_from: random cylinder separation in range from
	@param cylinder_sep_to: random cylinder separation in range to
	@return: nothing
	"""
	print 'Begin data generation with ' + str(no_iter) + ' iterations and ' + str(no_voxels) + ' in every iteration'
	for i in range(0, no_iter):
		print('Running iteration ' + str(i+1) + ' of ' + str(no_iter))
		print(time.strftime("%c"))
		dirname = "./data/gen/" + str(i) + '-' + str(datetime.datetime.now().isoformat()) + '/'

		# Sample cylinder radius and separation in range
		radius = (cylinder_rad_to - cylinder_rad_from) * np.random.random_sample() + cylinder_rad_from
		separation = (cylinder_sep_to - cylinder_sep_from) * np.random.random_sample() + cylinder_sep_from

		# Get a config file from describing the generated data
		config = _get_config(voxels=no_voxels, cylinder_rad=radius, cylinder_sep=separation, dir_name=dirname)
		# Perform the actual data generation
		_generate_data(config)


# Helper method to call bash script
def _generate_data(config):
	if not os.path.exists(config['dir_name']):
		try:
			os.makedirs(config['dir_name'])
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise

	# Write the config file
	with open(config['dir_name'] + 'config.json', 'w') as outfile:
		json.dump(config, outfile, sort_keys=True, indent=4)

	# Run camino and generate data
	p = subprocess.Popen(['./run_camino.sh', str(config['walkers']), str(config['tmax']), str(config['voxels']), str(config['p']), str(config['scheme_path']), str(config['cylinder_rad']), str(config['cylinder_sep']), str(config['out_name'])], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	p.communicate()

	# Write the target file
	rad = np.full(config['voxels'], config['cylinder_rad'])
	sep = np.full(config['voxels'], config['cylinder_sep'])
	np.savetxt(config['dir_name'] + 'targets.txt', np.transpose([rad, sep]))


# Generates a config about what is generated
def _get_config(walkers=100000, tmax=1000, voxels=1, p=0.0, scheme_path='./data/hpc.scheme', cylinder_rad=1E-6, cylinder_sep=2.1E-6, dir_name=''):
	out_name = str(dir_name) + 'cylinders.bfloat'
	obj = {
		'walkers': walkers,
		'tmax': tmax,
		'voxels': voxels,
		'p': p,
		'scheme_path': scheme_path,
		'cylinder_rad': cylinder_rad,
		'cylinder_sep': cylinder_sep,
		'dir_name': dir_name,
		'out_name': out_name
	}
	return obj


if __name__ == '__main__':
	run()
