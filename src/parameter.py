from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import degree
from rich.progress import track
from torch_geometric.datasets import TUDataset
import networkx as nx
import networkx.exception as nx_exception


class GraphParameters:
    def __init__(self, dataset_name, verbose=False):
        self.dataset = TUDataset(root='data/TUDataset',
                                 name=dataset_name)
        self.verbose = verbose
    
    def get_average_degree(self):
        '''
        Calculate the average degree of the graph
        '''
        degs = []

        for data in track(self.dataset, description=f'Calculating average degree for {self.dataset.name}', disable=not self.verbose):
            deg = degree(data.edge_index[0], data.num_nodes)
            degs.append(deg.mean().item())

        return sum(degs) / len(degs)
    
    def get_average_shortest_path(self):
        '''
        Calculate the average shortest path of the graph
        '''
        avg_shortest_paths = []

        for data in track(
            self.dataset,
            description=f'Calculating average shortest path for {self.dataset.name}',
            disable=not self.verbose
        ):
            G = to_networkx(data)
            if G.is_directed():
                G = G.to_undirected()
            for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
                if len(C) <= 1:
                    continue
                avg_shortest_paths.append(nx.average_shortest_path_length(C))

        return sum(avg_shortest_paths) / len(avg_shortest_paths)
    
    def get_graph_diameter(self):
        '''
        Calculate the diameter of the graph
        '''
        diameters = []

        for data in track(
                self.dataset,
                description=f'Calculating graph diameter for {self.dataset.name}',
                disable=not self.verbose):
            G = to_networkx(data)

            if G.is_directed():
                G = G.to_undirected()
            for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
                if len(C) <= 1:
                    continue
                try:
                    diameters.append(nx.diameter(C))
                except nx_exception.NetworkXError:
                    pass

        return sum(diameters) / len(diameters)

    def get_graph_density(self):
        '''
        Calculate the density of the graph
        '''
        densities = []

        for data in track(
                self.dataset,
                description=f'Calculating graph density for {self.dataset.name}',
                disable=not self.verbose):
            G = to_networkx(data)
            densities.append(nx.density(G))

        return sum(densities) / len(densities)
    
    def get_graph_clustering_coefficient(self):
        '''
        Calculate the clustering coefficient of the graph
        '''
        clustering_coefficients = []

        for data in track(
            self.dataset,
            description=f'Calculating graph clustering coefficient for {self.dataset.name}',
            disable=not self.verbose

        ):
            G = to_networkx(data)
            clustering_coefficients.append(nx.average_clustering(G))

        return sum(clustering_coefficients) / len(clustering_coefficients)


    def get_graph_transitivity(self):
        '''
        Calculate the transitivity of the graph
        '''
        transitivity = []

        for data in track(
                self.dataset,
                description=f'Calculating graph transitivity for {self.dataset.name}',
                disable=not self.verbose):
            G = to_networkx(data)
            transitivity.append(nx.transitivity(G))

        return sum(transitivity) / len(transitivity)
    
    def get_graph_assortativity(self):
        '''
        Calculate the assortativity of the graph
        '''
        assortativity = []

        for data in track(
                self.dataset,
                description=f'Calculating graph assortativity for {self.dataset.name}',
                disable=not self.verbose):
            G = to_networkx(data)
            assortativity.append(nx.degree_assortativity_coefficient(G))

        return sum(assortativity) / len(assortativity)
    
    def get_average_closeness_centrality(self):
        '''
        Calculate the average closeness centrality of the graph
        '''
        closeness_centralities = []

        for data in track(
                self.dataset,
                description=f'Calculating average closeness centrality for {self.dataset.name}',
                disable=not self.verbose):
            G = to_networkx(data)
            closeness_centralities.append(sum(nx.closeness_centrality(G).values())
                                        / G.number_of_nodes())

        return sum(closeness_centralities) / len(closeness_centralities)
    
    def get_average_betweenness_centrality(self):
        '''
        Calculate the average betweenness centrality of the graph
        '''
        betweenness_centralities = []

        for data in track(
                self.dataset,
                description=f'Calculating average betweenness centrality for {self.dataset.name}',
                disable=not self.verbose):
            G = to_networkx(data)
            betweenness_centralities.append(sum(nx.betweenness_centrality(G).values())
                                            / G.number_of_nodes())

        return sum(betweenness_centralities) / len(betweenness_centralities)
    
    def get_average_eigenvector_centrality(self):
        '''
        Calculate the average eigenvector centrality of the graph, given the dataset name
        '''
        eigenvector_centralities = []

        for data in track(
                self.dataset,
                description=f'Calculating average eigenvector centrality for {self.dataset.name}',
                disable=not self.verbose):
            G = to_networkx(data)
            eigenvector_centralities.append(sum(nx.eigenvector_centrality(
                G, max_iter=100000).values()) / G.number_of_nodes())

        return sum(eigenvector_centralities) / len(eigenvector_centralities)
    
    def wl_1d_color_count(self):
        '''
        Calculate the average number of coloring in 1WL of the graph
        '''
        color_count_sum = []

        for data in track(
                self.dataset,
                description=f'Calculating average number of coloring in 1WL for {self.dataset.name}',
                disable=not self.verbose):
            Graph = to_networkx(data)
            # Initialize all nodes with the same color
            colors = {node: 0 for node in Graph.nodes()}
            stable = False

            # Perform color refinement
            while not stable:
                new_colors = {}
                for node in Graph.nodes():
                    neighbor_colors = sorted([colors[neighbor] for neighbor
                                            in Graph.neighbors(node)])
                    new_colors[node] = f"{colors[node]}|{
                        '|'.join(map(str, neighbor_colors))}"
                label_to_color = {}
                for new_label in new_colors.values():
                    if new_label not in label_to_color:
                        label_to_color[new_label] = len(label_to_color)

                # Apply the new coloring
                new_colors = {node: label_to_color[new_colors[node]] for node
                            in Graph.nodes()}

                # Check for stability
                if colors == new_colors:
                    stable = True
                colors = new_colors

            color_count_sum.append(len(set(colors.values())))

        return sum(color_count_sum) / len(color_count_sum)


def get_average_degree(dataset_name, verbose=False):
    '''
    Calculate the average degree of the graph
    '''
    dataset = TUDataset(root='data/TUDataset',
                        name=dataset_name, use_node_attr=True)
    degs = []

    for data in track(dataset, description=f'Calculating average degree for {dataset_name}', disable=not verbose):
        deg = degree(data.edge_index[0], data.num_nodes)
        degs.append(deg.mean().item())

    return sum(degs) / len(degs)


def get_average_shortest_path(dataset_name, show_errors=False, verbose=False):
    '''
    Calculate the average shortest path of the graph
    '''
    dataset = TUDataset(root='data/TUDataset', name=dataset_name,
                        use_node_attr=True)
    avg_shortest_paths = []
    num_errors = 0

    for data in track(
        dataset,
        description=f'Calculating average shortest path for {dataset_name}',
        disable=not verbose
    ):
        G = to_networkx(data)
        if G.is_directed():
            G = G.to_undirected()
        for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
            if len(C) <= 1:
                continue
            avg_shortest_paths.append(nx.average_shortest_path_length(C))

            # print the graph
    if num_errors > 0:
        print(f'Number of errors: {num_errors}', end='')
        print(f'Error rate: {num_errors / len(dataset)}')

    return sum(avg_shortest_paths) / len(avg_shortest_paths)


def get_graph_diameter(dataset_name, show_errors=False, verbose=False):
    '''
    Calculate the diameter of the graph
    '''
    dataset = TUDataset(root='data/TUDataset',
                        name=dataset_name, use_node_attr=True)
    diameters = []
    num_errors = 0

    for data in track(
            dataset,
            description=f'Calculating graph diameter for {dataset_name}',
            disable=not verbose):
        G = to_networkx(data)

        if G.is_directed():
            G = G.to_undirected()
        for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
            if len(C) <= 1:
                continue
            try:
                diameters.append(nx.diameter(C))
            except nx_exception.NetworkXError:
                num_errors += 1
    if num_errors > 0:
        print(f'Number of errors: {num_errors}')
        print(f'Error rate: {num_errors / len(dataset)}')

    return sum(diameters) / len(diameters)


def get_graph_density(dataset_name, verbose=False):
    '''
    Calculate the density of the graph
    '''
    dataset = TUDataset(root='data/TUDataset',
                        name=dataset_name, use_node_attr=True)
    densities = []

    for data in track(
            dataset,
            description=f'Calculating graph density for {dataset_name}',
            disable=not verbose):
        G = to_networkx(data)
        densities.append(nx.density(G))

    return sum(densities) / len(densities)


def get_graph_clustering_coefficient(dataset_name, verbose=False):
    '''
    Calculate the clustering coefficient of the graph
    '''
    dataset = TUDataset(root='data/TUDataset',
                        name=dataset_name, use_node_attr=True)
    clustering_coefficients = []

    for data in track(
        dataset,
        description=f'Calculating graph clustering coefficient for {
            dataset_name}',
        disable=not verbose

    ):
        G = to_networkx(data)
        clustering_coefficients.append(nx.average_clustering(G))

    return sum(clustering_coefficients) / len(clustering_coefficients)


def get_graph_transitivity(dataset_name, verbose=False):
    '''
    Calculate the transitivity of the graph
    '''

    dataset = TUDataset(root='data/TUDataset',
                        name=dataset_name, use_node_attr=True)
    transitivity = []

    for data in track(
            dataset,
            description=f'Calculating graph transitivity for {dataset_name}',
            disable=not verbose):
        G = to_networkx(data)
        transitivity.append(nx.transitivity(G))

    return sum(transitivity) / len(transitivity)


def get_graph_assortativity(dataset_name, verbose=False):
    '''
    Calculate the assortativity of the graph
    '''
    dataset = TUDataset(root='data/TUDataset',
                        name=dataset_name, use_node_attr=True)
    assortativity = []

    for data in track(
            dataset,
            description=f'Calculating graph assortativity for {dataset_name}',
            disable=not verbose):
        G = to_networkx(data)
        assortativity.append(nx.degree_assortativity_coefficient(G))

    return sum(assortativity) / len(assortativity)


def get_average_closeness_centrality(dataset_name, verbose=False):
    '''
    Calculate the average closeness centrality of the graph
    '''
    dataset = TUDataset(root='data/TUDataset',
                        name=dataset_name, use_node_attr=True)
    closeness_centralities = []

    for data in track(
            dataset,
            description=f'Calculating average closeness centrality for {
                dataset_name}',
            disable=not verbose):
        G = to_networkx(data)
        closeness_centralities.append(sum(nx.closeness_centrality(G).values())
                                      / G.number_of_nodes())

    return sum(closeness_centralities) / len(closeness_centralities)


def get_average_betweenness_centrality(dataset_name, verbose=False):
    '''
    Calculate the average betweenness centrality of the graph
    '''
    dataset = TUDataset(root='data/TUDataset',
                        name=dataset_name, use_node_attr=True)
    betweenness_centralities = []

    for data in track(
            dataset,
            description=f'Calculating average betweenness centrality for {
                dataset_name}',
            disable=not verbose):
        G = to_networkx(data)
        betweenness_centralities.append(sum(nx.betweenness_centrality(G).values())
                                        / G.number_of_nodes())

    return sum(betweenness_centralities) / len(betweenness_centralities)


def get_average_eigenvector_centrality(dataset_name, verbose=False):
    '''
    Calculate the average eigenvector centrality of the graph, given the dataset name
    '''
    dataset = TUDataset(root='data/TUDataset',
                        name=dataset_name, use_node_attr=True)
    eigenvector_centralities = []

    for data in track(
            dataset,
            description=f'Calculating average eigenvector centrality for {
                dataset_name}',
            disable=not verbose):
        G = to_networkx(data)
        eigenvector_centralities.append(sum(nx.eigenvector_centrality(
            G, max_iter=100000).values()) / G.number_of_nodes())

    return sum(eigenvector_centralities) / len(eigenvector_centralities)


def wl_1d_color_count(dataset_name, verbose=False):
    '''
    Calculate the average number of coloring in 1WL of the graph
    '''
    dataset = TUDataset(root='data/TUDataset',
                        name=dataset_name, use_node_attr=True)
    color_count_sum = []

    for data in track(
            dataset,
            description=f'Calculating average number of coloring in 1WL for {
                dataset_name}',
            disable=not verbose):
        Graph = to_networkx(data)
        # Initialize all nodes with the same color
        colors = {node: 0 for node in Graph.nodes()}
        stable = False

        # Perform color refinement
        while not stable:
            new_colors = {}
            for node in Graph.nodes():
                neighbor_colors = sorted([colors[neighbor] for neighbor
                                          in Graph.neighbors(node)])
                new_colors[node] = f"{colors[node]}|{
                    '|'.join(map(str, neighbor_colors))}"
            label_to_color = {}
            for new_label in new_colors.values():
                if new_label not in label_to_color:
                    label_to_color[new_label] = len(label_to_color)

            # Apply the new coloring
            new_colors = {node: label_to_color[new_colors[node]] for node
                          in Graph.nodes()}

            # Check for stability
            if colors == new_colors:
                stable = True
            colors = new_colors

        color_count_sum.append(len(set(colors.values())))

    return sum(color_count_sum) / len(color_count_sum)
