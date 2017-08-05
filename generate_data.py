import subprocess
import os
import errno
import json
import numpy as np
import time

np.random.seed(int(round(time.time())))


def run(begin=20, no_iter=10, no_voxels=1000, cylinder_rad_from=1E-8, cylinder_rad_to=1E-6, cylinder_sep_from=2.1E-6, cylinder_sep_to=2.1E-6):
	print 'Begin data generation with ' + str(no_iter) + ' iterations and ' + str(no_voxels) + ' in every iteration'
	for i in range(begin, begin + no_iter):
		print('Running iteration ' + str(i+1 - begin) + ' of ' + str(no_iter))
		print(time.strftime("%c"))
		dirname = "./data/gen/" + str(i) + '/'

		radius = (cylinder_rad_to - cylinder_rad_from) * np.random.random_sample() + cylinder_rad_from
		separation = (cylinder_sep_to - cylinder_sep_from) * np.random.random_sample() + cylinder_sep_from

		config = _get_config(voxels=no_voxels, cylinder_rad=radius, cylinder_sep=separation, dir_name=dirname)
		_generate_data(config)


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
	out, err = p.communicate()

	# Write the target file
	rad = np.full(config['voxels'], config['cylinder_rad'])
	sep = np.full(config['voxels'], config['cylinder_sep'])
	np.savetxt(config['dir_name'] + 'targets.txt', np.transpose([rad, sep]))


def _get_config(walkers=100000, tmax=1000, voxels=1, p=0.0, scheme_path='hpc.scheme', cylinder_rad=1E-6, cylinder_sep=2.1E-6, dir_name=''):
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
