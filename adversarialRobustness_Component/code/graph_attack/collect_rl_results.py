import os

if __name__ == '__main__':
	result_root = '../../dropbox/scratch/results/graph_classification/components'
	targets = os.listdir(result_root)
	targets = sorted(targets)
	for fname in targets:
                if fname[0] == '.':
                    continue
		configs = os.listdir(result_root + '/' + fname)
		best_num = 100
		best_config = None

		for config in configs:
                        if config[0] == '.' or 'epoch-best' in config:
                            continue
			if '0.1' in config:
				continue
			result = result_root + '/' + fname + '/' + config + '/epoch-best.txt'
			with open(result, 'r') as f:
				num = float(f.readline().strip())
			if num < best_num:
				best_config = config
				best_num = num
		print fname, best_config, best_num	
