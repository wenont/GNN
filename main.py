from train_procedure import get_generalization_error_from_a_dataset
import pandas as pd
import logging
from tabulate import tabulate
from utils import get_average_degree, get_average_shortest_path, read_file_to_list
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt


def calculate_generalation_error():
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='./results/parameters.log', encoding='utf-8', level=logging.INFO, format='%(message)s', filemode='a')
    datasets = read_file_to_list('datasets.txt')
    
    df = pd.DataFrame({
        'Name': [],
        'Ave. generalization error': [],
        'std': []
    })
    
    for dataset in datasets:
        generalization_error, std = get_generalization_error_from_a_dataset(dataset)
        df.loc[len(df)] = [dataset, generalization_error, std]
    
    logger.info(tabulate(df, headers='keys', tablefmt='psql'))
    df.to_csv('generalization_error.csv')

def calcualte_parameters():
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='./results/parameters.log', encoding='utf-8', level=logging.INFO, format='%(message)s', filemode='a')
    datasets = read_file_to_list('datasets.txt')

    df = pd.DataFrame({
        'Name': [],
        'Ave. degree': [],
        'Ave. shortest path': [],
    })

    for dataset in tqdm(datasets):
        df.loc[len(df)] = [dataset, get_average_degree(dataset), get_average_shortest_path(dataset)]

    logger.info(tabulate(df, headers='keys', tablefmt='psql'))
    df.to_csv('parameters.csv')


def compare_generalization_error_and_parameters():

    df1 = pd.read_csv('results/generalization_error.csv')
    df2 = pd.read_csv('results/parameters.csv')

    df_combined = pd.merge(df1, df2, on='Name')

    df_sorted_by_degree = df_combined.sort_values('Ave. degree')
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df_sorted_by_degree['Ave. degree'], df_sorted_by_degree['Ave. generalization error'], marker='o', color='b')
    plt.title('Ave. Degree vs. Ave. Generalization Error')
    plt.xlabel('Ave. Degree')
    plt.ylabel('Ave. Generalization Error')

    df_sorted_by_shortest_path = df_combined.sort_values('Ave. shortest path')
    plt.subplot(1, 2, 2)
    plt.plot(df_sorted_by_shortest_path['Ave. shortest path'], df_sorted_by_shortest_path['Ave. generalization error'], marker='o', color='r')
    plt.title('Ave. Shortest Path vs. Ave. Generalization Error')
    plt.xlabel('Ave. Shortest Path')
    plt.ylabel('Ave. Generalization Error')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # get input from user
    print('Please choose from the following options:')
    print('1. Get generalization error')
    print('2. Get Parameters')
    print('3. Compare generalization error and parameters')
    print('4. Exit')
    option = input('Enter your choice: ')

    if option == '1':
        calculate_generalation_error()
    elif option == '2':
        calcualte_parameters()
    elif option == '3':
        compare_generalization_error_and_parameters()