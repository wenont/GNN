from train import get_generalization_error_from_a_dataset
import pandas as pd
import logging
from tabulate import tabulate
from utils import (
    get_average_degree,
    get_average_shortest_path,
    get_graph_diameter,
    get_graph_density,
    get_graph_clustering_coefficient,
    get_average_closeness_centrality,
    get_average_betweenness_centrality,
    get_average_eigenvector_centrality,
    wl_1d_color_count,
    read_file_to_list
)
import logging
import matplotlib.pyplot as plt
import argparse

# from rich import print, Panel
import wandb
import os
import os.path as osp


parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument("-d", "--dataset", help="dataset name")
parser.add_argument("-f", "--function", help="function name")
args = parser.parse_args()

if args.dataset:
    DATAPATH = f'data/{args.dataset}.txt'
else:
    DATAPATH = 'data/test_dataset.txt'


def calculate_generalation_error():
    pass


def calcualte_parameters():
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='./log/parameters.log', encoding='utf-8',
                        level=logging.INFO, format='%(message)s', filemode='a')
    datasets = read_file_to_list(DATAPATH)

    # Initialize the DataFrame with all the required columns
    df = pd.DataFrame({
        'Name': [],
        'Ave. degree': [],
        'Ave. shortest path': [],
        'Graph diameter': [],
        'Graph density': [],
        'Graph clustering coefficient': [],
        'Ave. closeness centrality': [],
        'Ave. betweenness centrality': [],
        'Ave. eigenvector centrality': [],
        '1-WL color count': []
    })

    # Populate the DataFrame with data from each dataset
    len_datasets = len(datasets)
    for i, dataset in enumerate(datasets):
        print(Panel(f'[{i+1}/{len_datasets}]: [red]{dataset}'))
        df.loc[len(df)] = [
            dataset,
            get_average_degree(dataset, args.verbose),
            get_average_shortest_path(dataset, args.verbose),
            get_graph_diameter(dataset, args.verbose),
            get_graph_density(dataset, args.verbose),
            get_graph_clustering_coefficient(dataset, args.verbose),
            get_average_closeness_centrality(dataset, args.verbose),
            get_average_betweenness_centrality(dataset, args.verbose),
            get_average_eigenvector_centrality(dataset, args.verbose),
            wl_1d_color_count(dataset, args.verbose),
        ]

    logger.info(tabulate(df, headers='keys', tablefmt='psql'))
    df.to_csv(f'results/parameters_{args.dataset}.csv')


def compare_generalization_error_and_parameters():

    df1 = pd.read_csv('results/generalization_error.csv')
    df2 = pd.read_csv('results/parameters.csv')

    df_combined = pd.merge(df1, df2, on='Name')

    df_sorted_by_degree = df_combined.sort_values('Ave. degree')
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(df_sorted_by_degree['Ave. degree'],
                df_sorted_by_degree['Ave. generalization error'],
                marker='o', color='b')
    plt.title('Ave. Degree vs. Ave. Generalization Error')
    plt.xlabel('Ave. Degree')
    plt.ylabel('Ave. Generalization Error')

    df_sorted_by_shortest_path = df_combined.sort_values('Ave. shortest path')
    plt.subplot(1, 2, 2)
    plt.scatter(df_sorted_by_shortest_path['Ave. shortest path'],
                df_sorted_by_shortest_path['Ave. generalization error'],
                marker='o', color='r')
    plt.title('Ave. Shortest Path vs. Ave. Generalization Error')
    plt.xlabel('Ave. Shortest Path')
    plt.ylabel('Ave. Generalization Error')

    plt.tight_layout()
    plt.show()


def get_correlation():
    df1 = pd.read_csv('results/generalization_error.csv')
    df2 = pd.read_csv('results/parameters.csv')

    # correlation of ave. degree and ave. generalization error
    correlation1 = df1['Ave. generalization error'].corr(df2['Ave. degree'])
    print(f'Correlation of ave. degree and ave. generalization error: \
          {correlation1}')

    # correlation of ave. shortest path and ave. generalization error
    correlation2 = df1['Ave. generalization error'].corr(
        df2['Ave. shortest path'])
    print(f'Correlation of ave. shortest path and ave. generalization error: {
          correlation2}')


def interactive_mode():
    print('Please choose from the following options:')
    print('1. Get generalization error')
    print('2. Get Parameters')
    print('3. Compare generalization error and parameters')
    print('4. Get correlation')
    print('5. foo')
    print('6. Exit')
    option = input('Enter your choice: ')
    handle_option(option)


def get_best_hyperparameters(project_name: str = 'bt_GCN'):
    # Get hyperparameters from wandb
    api = wandb.Api()

    # Get the set of sweep id
    dataset_csv_path = osp.join(osp.dirname(
        __file__), '..', 'data', 'dataset.csv')
    df = pd.read_csv(dataset_csv_path)
    sweep_ids = df[df['project'] == project_name]['sweep_id'].values

# {'batch_size': 64, 'hidden_size': 256, 'dataset_name': 'MCF-7', 'normlization': 'graph', 'learning_rate': 0.006434620974325459, 'default_patience': 100, 'patience_plateau': 30, 'num_hidden_layers': 6}
    df_hyperparameters = pd.DataFrame({
        'sweep_id': [],
        'model_name': [],  # use project name
        'batch_size': [],
        'hidden_size': [],
        'dataset_name': [],
        'normlization': [],
        'learning_rate': [],
        'default_patience': [],
        'patience_plateau': [],
        'num_hidden_layers': []
    })

    for sweep_id in sweep_ids:
        sweep = api.sweep(
            f'wensz-rwth-aachen-university/{project_name}/{sweep_id}')
        runs = sorted(sweep.runs,
                      key=lambda run: run.summary.get("best_test_acc", 0), reverse=True)
        best_test_acc = runs[0].summary.get("best_test_acc", 0)
        print(f"Best run {runs[0].name} with {
              best_test_acc}% validation accuracy")

        # get the best hyperparameters
        best_hyperparameters = runs[0].config
        # save the best hyperparameters to dataframe
        df_hyperparameters.loc[len(df_hyperparameters)] = [
            sweep_id,
            project_name,
            best_hyperparameters['batch_size'],
            best_hyperparameters['hidden_size'],
            best_hyperparameters['dataset_name'],
            best_hyperparameters['normlization'],
            best_hyperparameters['learning_rate'],
            best_hyperparameters['default_patience'],
            best_hyperparameters['patience_plateau'],
            best_hyperparameters['num_hidden_layers']
        ]

    # save the best hyperparameters to csv, and print the table, do not replace the existing file
    df_hyperparameters.to_csv('results/best_hyperparameters.csv', mode='a')
    print(tabulate(df_hyperparameters, headers='keys', tablefmt='psql'))


def handle_option(option):
    if option == '1' or option == 'get_generalization_error':
        calculate_generalation_error()
    elif option == '2' or option == 'parameter':
        calcualte_parameters()
    elif option == '3' or option == 'compare_generalization_error_and_parameters':
        compare_generalization_error_and_parameters()
    elif option == '4' or option == 'get_correlation':
        get_correlation()
    elif option == '5' or option == 'get_hyperparameters':
        get_best_hyperparameters()
    else:
        if option.isdigit() and int(option) == 6:
            print('Exiting...')
        else:
            print('Invalid option. Please try again.')
            interactive_mode()


if __name__ == '__main__':
    # Set default as interactive mode, user can choose the option
    # Otherwise, use args to run the specific function
    if args.function:
        handle_option(args.function)
    else:
        interactive_mode()
