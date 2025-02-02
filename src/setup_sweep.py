from utils import read_file_to_list, setup_wandb_sweep
import os.path as osp
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--temporory", help="set a temporary flag",
                    action="store_true")
args = parser.parse_args()

print(f'Temporary flag: {args.temporory}')

project_name = 'bt_MPNN'
path = osp.join(osp.dirname(__file__), '..', 'data', 'test_dataset.txt')
dataset_list = read_file_to_list(path)

for dataset_name in dataset_list:
	print(f'Dataset name: {dataset_name}')
	path = osp.join(osp.dirname(__file__), '..', 'data', 'dataset.csv')
	df = pd.read_csv(path)

	# check if the df contains a row with the name of the dataset
	print('Checking if the sweep exists')
	if df[(df['name'] == dataset_name) & (df['project'] == project_name)].empty | args.temporory:
		print('Creating sweep')
		if args.temporory:
			sweep_id = setup_wandb_sweep(project_name, dataset_name, temp=True)
		else:
			sweep_id = setup_wandb_sweep(project_name, dataset_name)
		print(f'Sweep id: {sweep_id}')
		# save the dataset name, sweep_id and project name to the csv
		df = pd.concat([df, pd.DataFrame([{'name': f'{dataset_name}', 'project': project_name, 'sweep_id': sweep_id}])], ignore_index=True)
		df.to_csv(path, index=False)
	else:
		print('Dataset found')
		sweep_id = df[(df['name'] == dataset_name) & (df['project'] == project_name)]['sweep_id'].values[0]
		print(f'Sweep id: {sweep_id}')
