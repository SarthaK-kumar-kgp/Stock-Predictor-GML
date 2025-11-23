import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# from torch_geometric.utils import from_networkx
# import torch

class SectorGraph:
    def __init__(self, df):
        self.df = df.copy()
        self.features = None
        self.G = None
        self.data = None  # Will hold PyG object later

    def aggregate_features(self):
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Return'] = self.df.groupby('Ticker')['Close'].pct_change()

        self.features = self.df.groupby("Ticker").agg({
            "Open":"mean",
            "High":"mean",
            "Low":"mean",
            "Close":"mean",
            "Volume":"mean",
            "avg_sentiment":"mean",
            "headline_count":"mean",
            "VIX_Open":"mean",
            "VIX_High":"mean",
            "VIX_Low":"mean",
            "VIX_Close":"mean",
            "Return":"mean",
            "sector":"first"
        }).reset_index()

        return self.features

    def build_sector_graph(self):
        if self.features is None:
            self.aggregate_features()

        G = nx.Graph()

        for _, row in self.features.iterrows():
            G.add_node(
                row['Ticker'],
                Open=float(row['Open']), High=float(row['High']),
                Low=float(row['Low']), Close=float(row['Close']),
                Volume=float(row['Volume']),
                avg_sentiment=float(row['avg_sentiment']) if pd.notna(row['avg_sentiment']) else 0.0,
                headline_count=float(row['headline_count']) if pd.notna(row['headline_count']) else 0.0,
                VIX_Open=float(row['VIX_Open']), VIX_High=float(row['VIX_High']),
                VIX_Low=float(row['VIX_Low']), VIX_Close=float(row['VIX_Close']),
                Return=float(row['Return']) if pd.notna(row['Return']) else 0.0,
                sector=row['sector']
            )

        tickers = self.features['Ticker'].tolist()
        sectors = dict(zip(self.features['Ticker'], self.features['sector']))

        for i in range(len(tickers)):
            for j in range(i+1, len(tickers)):
                if pd.notna(sectors[tickers[i]]) and sectors[tickers[i]] == sectors[tickers[j]]:
                    G.add_edge(tickers[i], tickers[j])

        self.G = G
        return G

    # -------------------- VISUALIZATION -------------------- #

    def graph_stats(self):
        if self.G is None:
            self.build_sector_graph()
        print("Nodes:", self.G.number_of_nodes())
        print("Edges:", self.G.number_of_edges())
        print("Density:", nx.density(self.G))
        print("Avg Degree:", sum(dict(self.G.degree()).values())/self.G.number_of_nodes())
        print("Connected Components:", nx.number_connected_components(self.G))

    def print_edge_list(self, limit=30):
        if self.G is None:
            self.build_sector_graph()
        edges = list(self.G.edges())
        print(f"Total edges: {len(edges)}")
        print("Sample edges:", edges[:limit])

    def plot_graph(self, figsize=(10,8)):
        if self.G is None:
            self.build_sector_graph()

        sectors = nx.get_node_attributes(self.G, 'sector')
        sector_list = list(set(sectors.values()))
        color_map = {sec: i for i, sec in enumerate(sector_list)}
        node_colors = [color_map[sectors[n]] for n in self.G.nodes()]

        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.G, seed=42)
        nx.draw(self.G, pos, node_color=node_colors, with_labels=True, node_size=500)
        plt.title("Sector-wise Stock Graph")
        plt.show()

    def plot_adjacency_matrix(self):
        if self.G is None:
            self.build_sector_graph()
        A = nx.to_numpy_array(self.G)

        plt.figure(figsize=(8,6))
        sns.heatmap(A, cmap="viridis")
        plt.title("Adjacency Matrix")
        plt.xlabel("Node Index")
        plt.ylabel("Node Index")
        plt.show()
'''
    def to_pyg(self):
        if self.G is None:
            self.build_sector_graph()

        data = from_networkx(self.G)

        node_features = []
        for _, attrs in self.G.nodes(data=True):
            node_features.append([
                attrs['Open'], attrs['High'], attrs['Low'], attrs['Close'], attrs['Volume'],
                attrs['avg_sentiment'], attrs['headline_count'],
                attrs['VIX_Open'], attrs['VIX_High'], attrs['VIX_Low'], attrs['VIX_Close'],
                attrs['Return']
            ])

        data.x = torch.tensor(node_features, dtype=torch.float)
        self.data = data
        return data

'''
