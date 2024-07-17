import logging
from torch_geometric.datasets import TUDataset


logger = logging.getLogger(__name__)
logging.basicConfig(filename='dataset.log', encoding='utf-8', level=logging.DEBUG, format='%(message)s')

datasets = ['DD', 'ENZYMES', 'KKI', 'OHSU', 'Peking_1', 'PROTEINS', 'PROTEINS_full']


for name in datasets:
    logger.info(f'===================================={name}===================================')
    dataset = TUDataset(root='data/TUDataset', name=name)

    logger.info('====================')
    logger.info(f'Dataset: {dataset}:')
    logger.info('-'*20)
    logger.info(f'Number of graphs: {len(dataset)}')
    logger.info(f'Number of features: {dataset.num_features}')
    logger.info(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  


    logger.info('===========================================================')
    logger.info(f'{data}')
    logger.info('-'*60)
    logger.info(f'Number of nodes: {data.num_nodes}')
    logger.info(f'Number of edges: {data.num_edges}')
    logger.info(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    logger.info(f'Has isolated nodes: {data.has_isolated_nodes()}')
    logger.info(f'Has self-loops: {data.has_self_loops()}')
    logger.info(f'Is undirected: {data.is_undirected()}')

    logger.info('')
