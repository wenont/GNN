import os
# from train import get_generalization_error_from_a_dataset
import pandas as pd
import logging
# from tabulate import tabulate
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
    read_file_to_list,
    number_of_graphs
)
import logging
import matplotlib.pyplot as plt
import argparse

# from rich import print, Panel
import wandb
import os.path as osp
from utils import TrainParams

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


def calculate_generalation_error(file_path: str):
    if file_path is None:
        project_name = 'bt_SGC'
        file_path = osp.join(osp.dirname(__file__), 'results', f'best_hyperparameters_{project_name}.csv')
    best_hyperparameters = pd.read_csv(file_path)
    
    for i in range(len(best_hyperparameters)):
        dataset_name = best_hyperparameters.loc[i, 'dataset_name']
        model_name = best_hyperparameters.loc[i, 'model_name']
        batch_size = best_hyperparameters.loc[i, 'batch_size'].item()
        hidden_size = best_hyperparameters.loc[i, 'hidden_size'].item()
        normlization = best_hyperparameters.loc[i, 'normlization']
        learning_rate = best_hyperparameters.loc[i, 'learning_rate'].item()
        default_patience = best_hyperparameters.loc[i, 'default_patience'].item()
        patience_plateau = best_hyperparameters.loc[i, 'patience_plateau'].item()
        num_hidden_layers = best_hyperparameters.loc[i, 'num_hidden_layers'].item()

        # print(type(batch_size))
        # break
        # print(f"dataset_name: {dataset_name}, model_name: {model_name}, batch_size: {batch_size}, hidden_size: {hidden_size}, normlization: {normlization}, learning_rate: {learning_rate}, default_patience: {default_patience}, patience_plateau: {patience_plateau}, num_hidden_layers: {num_hidden_layers}")

        if model_name == 'bt_GCN':
            model_name = 'GCN'
        if model_name == 'bt_SGC':
            model_name = 'GCN'

        trainParams = TrainParams(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, batch_size=batch_size,
                                    learning_rate=learning_rate, patience_earlystopping=default_patience, patience_plateau=patience_plateau, normlization=normlization)

        generalization_error, standard_deviation = get_generalization_error_from_a_dataset(dataset_name=dataset_name, model_name=model_name, trainParams=trainParams)
        print(f"Generalization error for {dataset_name} is {generalization_error}")

        # Save the generalization error to a csv file
        df = pd.DataFrame({
            'Name': [dataset_name],
            'Ave. generalization error': [generalization_error],
            'Standard deviation': [standard_deviation]
        })
        df.to_csv(f'results/generalization_error_{project_name}.csv', mode='a', header=not os.path.exists(f'results/generalization_error_{project_name}.csv'))


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

    plt.tight_layout(rect=[0, 0, 1.5, 1])
    plt.show()


def get_correlation(model: str = 'SGC'):
    '''
    Get the correlation between the generalization error and the parameters
    1. Ave. degree
    2. Ave. shortest path
    3. Graph diameter
    4. Graph density
    5. Graph clustering coefficient
    6. Ave. closeness centrality
    7. Ave. betweenness centrality
    8. Ave. eigenvector centrality
    9. 1-WL color count

    print the correlation matrix

    Use scatter plot to show the correlation between the generalization error and the parameters
    '''

    # generalization_error_file = f'results/generalization_error_{model}.csv'
    generalization_error_file = f'results/output_{model}.csv'


    df1 = pd.read_csv(generalization_error_file)
    df2 = pd.read_csv('results/parameters.csv')

    df_combined = pd.merge(df1, df2, on='Name').drop(columns=['Unnamed: 0'])

    # Use scatter plot to show the correlation between the generalization error and the parameters
    # Use the function number_of_graphs to get the number of graphs
    # if the number of graphs is less than 1000, show the point in the scatter plot with color blue
    # if the number of graphs is more than 1000 but less than 4000, show the point in the scatter plot with color green
    # if the number of graphs is more than 4000, show the point in the scatter plot with color red
    num_columns = len(df_combined.drop(columns=['Name', 'Standard deviation']).columns)
    num_rows = (num_columns + 2) // 3
    plt.figure(figsize=(10, 12))
    for i, column in enumerate(df_combined.drop(columns=['Name', 'Standard deviation']).columns):
        if i == 0:
            continue
        plt.subplot(num_rows, 3, i)
        colors = []
        for name in df_combined['Name']:
            num_graphs = number_of_graphs(name)
            if num_graphs < 1000:
                colors.append('blue')
            elif num_graphs < 4000:
                colors.append('green')
            else:
                colors.append('red')
        scatter = plt.scatter(df_combined[column], df_combined['Ave. generalization error'], c=colors)
        correlation = df_combined[column].corr(df_combined['Ave. generalization error'])
        plt.title(f'Correlation: {correlation:.2f}')
        plt.xlabel(column)
        plt.ylabel('Ave. Generalization Error')
    
    # Create a legend for the whole plot
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='< 1000 graphs'),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='1000-4000 graphs'),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='> 4000 graphs')]
    plt.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.3), title='Size of the dataset')
    plt.subplots_adjust(wspace=0.3, hspace=40)
    plt.tight_layout()
    plt.show()
    
    # save the scatter plot to a file
    plt.savefig(f'results/correlation_{model}.png')

    # ignore the dataset with less than 1000 graphs
    # Use the function number_of_graphs to get the number of graphs
    df_combined_ignore_less_than_1000 = df_combined.copy()
    for i, name in enumerate(df_combined_ignore_less_than_1000['Name']):
        num_graphs = number_of_graphs(name)
        if num_graphs < 1000:
            df_combined_ignore_less_than_1000.drop(i, inplace=True)
    num_columns = len(df_combined_ignore_less_than_1000.drop(columns= ['Name', 'Standard deviation']).columns)
    num_rows = (num_columns + 2) // 3
    plt.figure(figsize=(10, 10))
    for i, column in enumerate(df_combined_ignore_less_than_1000.drop(columns= ['Name', 'Standard deviation']).columns):
        if i == 0:
            continue
        plt.subplot(num_rows, 3, i)
        plt.scatter(df_combined_ignore_less_than_1000[column], df_combined_ignore_less_than_1000['Ave. generalization error'])
        correlation = df_combined_ignore_less_than_1000[column].corr(df_combined_ignore_less_than_1000['Ave. generalization error'])
        plt.title(f'Correlation: {correlation:.2f}')
        plt.xlabel(column)
        plt.ylabel('Ave. Generalization Error')
        # Create a legend for the whole plot
    plt.tight_layout()
    # save the scatter plot to a file
    plt.savefig(f'results/correlation_ignore_less_than_1000_{model}.png')


def interactive_mode():
    print('Please choose from the following options:')
    print('1. Get generalization error')
    print('2. Get Parameters')
    print('3. Compare generalization error and parameters')
    print('4. Get correlation')
    print('5. Get best hyperparameters')
    print('6. Sum the parameters')
    print('7. Exit')
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

    if project_name == 'bt_GATv2':
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
            'num_hidden_layers': [],
            'heads': [],
            'dropout': [],
            'residual': [],
        })
    else:
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

        if project_name == 'bt_GATv2':
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
                best_hyperparameters['num_hidden_layers'],
                best_hyperparameters['heads'],
                best_hyperparameters['dropout'],
                best_hyperparameters['residual'],
            ]
        else:
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
    df_hyperparameters.to_csv(f'results/best_hyperparameters_{project_name}.csv', mode='a')
    # print(tabulate(df_hyperparameters, headers='keys', tablefmt='psql'))


def sum_the_parameters():
    # read the files in the path ../results/, the file names are like parameters_*.csv
    # sum the parameters of each dataset
    # save the results to a csv file
    path = osp.join(osp.dirname(__file__), '..', 'results')
    files = os.listdir(path)
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

    for file in files:
        if file.startswith('parameters_'):
            df_temp = pd.read_csv(osp.join(path, file))
            df = pd.concat([df, df_temp], ignore_index=True)
        
    df_sum = pd.DataFrame({
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

    df_sum.loc[0] = [
        'Sum',
        df['Ave. degree'].sum(),
        df['Ave. shortest path'].sum(),
        df['Graph diameter'].sum(),
        df['Graph density'].sum(),
        df['Graph clustering coefficient'].sum(),
        df['Ave. closeness centrality'].sum(),
        df['Ave. betweenness centrality'].sum(),
        df['Ave. eigenvector centrality'].sum(),
        df['1-WL color count'].sum()
    ]

    df = pd.concat([df, df_sum], ignore_index=True)
    df = df.drop(columns=['Unnamed: 0'])
    df.to_csv('results/sum_parameters.csv', index=False)


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
        project_name = input('Enter the project name: ')
        get_best_hyperparameters(project_name)
    elif option == '6' or option == 'sum_the_parameters':
        sum_the_parameters()
    else:
        if option.isdigit() and int(option) == 6:
            print('Exiting...')
        else:
            print('Invalid option. Please try again.')
            interactive_mode()


if __name__ == '__main__':
    # # Set default as interactive mode, user can choose the option
    # # Otherwise, use args to run the specific function
    # if args.function:
    #     handle_option(args.function)
    # else:
    #     interactive_mode()
    # model = 'GCN'
    # get_correlation(model)

    get_best_hyperparameters('bt_GATv2')
