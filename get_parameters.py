from utils import get_average_degree, get_average_shortest_path
import pandas as pd
from tqdm import tqdm
import logging
from tabulate import tabulate


logger = logging.getLogger(__name__)
logging.basicConfig(filename='./results/parameters.log', encoding='utf-8', level=logging.INFO, format='%(message)s', filemode='a')
datasets = ["FRANKENSTEIN", "NCI1", "COIL-RAG", "Letter-high", "DD", "PROTEINS_full", "COLORS-3"]

df = pd.DataFrame({
    'Name': [],
    'Ave. degree': [],
    'Ave. shortest path': [],
})


for dataset in tqdm(datasets):

    df.loc[len(df)] = [dataset, get_average_degree(dataset), get_average_shortest_path(dataset)]


logger.info(tabulate(df, headers='keys', tablefmt='psql'))
df.to_csv('parameters.csv')

    
    