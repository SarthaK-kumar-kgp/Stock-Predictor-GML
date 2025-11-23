
THRESHOLD = 0.7        # keep edge if |rho_ij| >= 0.7
MIN_OVERLAP = 0.80     # require >=80% common timestamps per pair
REMOVE_SELF_LOOPS = True
UNDIRECTED = True


class StockCorrGraph:
    """
    Build a fixed-spec correlation graph from price data (long format).
    Expects a long DataFrame with columns at least: ["Date","Ticker","Close"].

    Spec:
    * Returns = log returns
    * Correlation = Spearman (signed, raw)
    * Sparsify = threshold on |rho| with tau=0.7
    * Pairwise overlap >= 80% of rows
    * No self-loops; undirected (store src<dst)
    * Node IDs = alphabetical ticker order (0..N-1)

    Methods:
      - build()    -> (adj_list, edge_index, edge_weight, node_index_map)
      - testing()  -> diagnostics/plots; strictly calls self.build()
    """

    def __init__(self, price_df: pd.DataFrame):
        if not isinstance(price_df, pd.DataFrame):
            raise ValueError("price_df must be a pandas DataFrame with columns ['Date','Ticker','Close', ...].")

        # Wide pivot: Date x Ticker -> Close
        wide = price_df.pivot(index="Date", columns="Ticker", values="Close")
        wide = wide.sort_index()

        # Deterministic node order: alphabetical tickers
        tickers = sorted([str(c) for c in wide.columns])
        self.ticker_to_idx: Dict[str, int] = {t: i for i, t in enumerate(tickers)}

        # Reorder & clean
        df = wide[tickers].copy().sort_index()
        df = df.apply(pd.to_numeric, errors="coerce").ffill()  # forward-fill only

        self.prices = df              # wide: Date x Ticker
        self.tickers = tickers        # alphabetical tickers

    def build(self) -> Tuple[List[Tuple[int, int, float]], torch.LongTensor, torch.FloatTensor, Dict[str, int]]:
        """
        Returns:
          adj_list:    List[(int src, int dst, float weight)]
          edge_index:  torch.LongTensor shape (2, E)
          edge_weight: torch.FloatTensor shape (E,)
          node_index_map: Dict[str, int]  (ticker -> node id)
        """
        # Log returns
        rets = np.log(self.prices).diff().dropna(how="all")

        # Pairwise overlap requirement
        valid = rets.notna().astype(np.int32)            # 1 where present
        T = len(rets.index)
        min_common = math.ceil(MIN_OVERLAP * T)
        overlap_counts = valid.T @ valid                 # (N x N) integer matrix

        # Spearman correlation (pandas handles ranking)
        corr = rets.corr(method="spearman", min_periods=min_common)  # DataFrame (N x N)
        rho = corr.values

        # Overlap mask
        overlap_mask = overlap_counts.values >= min_common
        if REMOVE_SELF_LOOPS:
            np.fill_diagonal(overlap_mask, False)

        # Threshold on magnitude, keep raw signed weights
        mag = np.abs(rho)
        keep = (mag >= THRESHOLD) & overlap_mask

        # Undirected: enforce upper triangle to avoid duplicates
        if UNDIRECTED:
            keep = np.triu(np.logical_or(keep, keep.T), k=1)

        # Extract edges
        src, dst = np.where(keep)
        weights = rho[src, dst].astype(np.float32)

        # Build outputs
        adj_list: List[Tuple[int, int, float]] = [(int(i), int(j), float(w)) for i, j, w in zip(src, dst, weights)]
        edge_index = torch.tensor([src, dst], dtype=torch.long)         # (2, E)
        edge_weight = torch.tensor(weights, dtype=torch.float32)        # (E,)

        return adj_list, edge_index, edge_weight, self.ticker_to_idx
    def correlation_matrix(self) -> pd.DataFrame:
        """
        Return the full n x n Spearman correlation matrix (tickers x tickers),
        using the same MIN_OVERLAP logic as build().

        DOES NOT APPLY THE THRESHOLD FUNCTIONS
        DOES NOT APPLY EXTRA THRESHOLD MASK
        DOES NOT ENFORCE UPPER TRIANGLE MATRIX
        DOES NOT CONVERT TO EDGES/EDGE INDEX
        """
        # Log returns
        rets = np.log(self.prices).diff().dropna(how="all")

        # Overlap requirement
        valid = rets.notna().astype(np.int32)
        T = len(rets.index)
        min_common = math.ceil(MIN_OVERLAP * T)

        # Spearman correlation with overlap constraint
        corr = rets.corr(method="spearman", min_periods=min_common)

        # Just to be explicit: index/columns are tickers
        corr = corr.loc[self.tickers, self.tickers]
        return corr

    def testing(
        self,
        long_df: pd.DataFrame,
        focus_ticker: Optional[str] = None,             # e.g., "MMM"
        edge_pair: Optional[Tuple[str, str]] = None,    # e.g., ("MMM","HON")
        k_neighbors: int = 10,
        top_n_news_days: int = 5,
        draw_graph: bool = True,
        figsize: Tuple[int, int] = (10, 8),
        seed: int = 42,
    ) -> Dict[str, object]:
        """
        One-stop diagnostics for the correlation graph.


        - Draws the graph with weights (blue=+ρ, red=−ρ; width ∝ |ρ|)
        - Prints top-|ρ| neighbors for a focus ticker (defaults to highest-degree node)
        - Deep-dives one edge: normalized price series + returns scatter with Spearman ρ & p-value
        - Sector-sector matrix of mean edge weights (if sector metadata available)
        - Optional news/sentiment snapshot on extreme joint-move days (if columns available)

        Returns dict of artifacts:
          G, pos, adj_list, edge_index, edge_weight, node_index_map, idx_to_ticker,
          neighbors, edge_stats, sector_matrix, news_snapshot
        """
        # ---------- 1) Build graph via the class's build() ----------
        adj_list, edge_index, edge_weight, node_index_map = self.build()
        idx_to_ticker = {idx: t for t, idx in node_index_map.items()}

        # ---------- 2) Build NetworkX graph ----------
        G = nx.Graph()
        for u, v, w in adj_list:
            G.add_edge(u, v, weight=float(w))

        # ---------- 3) Optional drawing ----------
        pos = None
        if draw_graph and G.number_of_edges() > 0:
            weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
            edge_colors = ["tab:red" if w < 0 else "tab:blue" for w in weights]
            edge_widths = 1.0 + 3.0 * (np.abs(weights) - THRESHOLD) / (1.0 - THRESHOLD)
            edge_widths = np.clip(edge_widths, 1.0, 4.0)

            plt.figure(figsize=figsize)
            pos = nx.spring_layout(G, seed=seed)
            nx.draw_networkx_nodes(G, pos, node_size=120, node_color="lightgray",
                                   linewidths=0.5, edgecolors="k")
            nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.8)

            degree_sorted = sorted(G.degree, key=lambda x: x[1], reverse=True)[:25]
            labels_subset = {n: idx_to_ticker[n] for n, _ in degree_sorted}
            nx.draw_networkx_labels(G, pos, labels=labels_subset, font_size=8)

            plt.title(f"Stock Correlation Graph (Spearman; edges |rho| ≥ {THRESHOLD})")
            plt.axis("off")
            plt.tight_layout()
            plt.show()

        # ---------- 4) Metadata from long_df (latest sector/industry per ticker) ----------
        TICKER_TO_SECTOR: Dict[str, str] = {}
        TICKER_TO_INDUSTRY: Dict[str, str] = {}
        if isinstance(long_df, pd.DataFrame) and {"Date", "Ticker"}.issubset(long_df.columns):
            latest_meta = (long_df.sort_values("Date")
                                  .groupby("Ticker").tail(1)
                                  .set_index("Ticker"))
            if "sector" in latest_meta.columns:
                TICKER_TO_SECTOR = latest_meta["sector"].astype(str).to_dict()
            if "industry" in latest_meta.columns:
                TICKER_TO_INDUSTRY = latest_meta["industry"].astype(str).to_dict()

        # ---------- 5) Returns matrix (for diagnostics below) ----------
        rets = np.log(self.prices).diff().dropna(how="all")

        # ---------- 6) Neighbor inspection ----------
        neighbors_out = []
        if focus_ticker is None and len(G) > 0:
            focus_node = max(G.degree, key=lambda x: x[1])[0]
            focus_ticker = idx_to_ticker[focus_node]

        if focus_ticker is not None and focus_ticker in node_index_map:
            u = node_index_map[focus_ticker]
            nbrs = []
            for v in G.neighbors(u):
                w = G[u][v]["weight"]
                tkr_v = idx_to_ticker[v]
                sec = TICKER_TO_SECTOR.get(tkr_v, "?")
                ind = TICKER_TO_INDUSTRY.get(tkr_v, "?")
                nbrs.append((tkr_v, float(w), sec, ind))
            nbrs = sorted(nbrs, key=lambda x: abs(x[1]), reverse=True)[:k_neighbors]
            print(f"\nNeighbors of {focus_ticker} (top {k_neighbors} by |rho|):")
            for tkr_v, w, sec, ind in nbrs:
                print(f"  {tkr_v:8s}  rho={w:+.3f}  sector={sec:>14s}  industry={ind}")
            neighbors_out = nbrs

        # ---------- 7) Edge deep-dive ----------
        edge_stats = None
        if edge_pair is None and focus_ticker is not None and neighbors_out:
            edge_pair = (focus_ticker, neighbors_out[0][0])

        if edge_pair is not None:
            ti, tj = edge_pair
            if ti in self.tickers and tj in self.tickers:
                s1 = self.prices[ti].dropna()
                s2 = self.prices[tj].dropna()
                both = pd.concat([s1, s2], axis=1, join="inner").dropna()
                both.columns = [ti, tj]

                r1 = rets[ti].dropna()
                r2 = rets[tj].dropna()
                rboth = pd.concat([r1, r2], axis=1, join="inner").dropna()
                rboth.columns = [ti, tj]

                rho, p = spearmanr(rboth[ti], rboth[tj])
                print(f"\nEdge {ti} — {tj}: Spearman rho={rho:+.3f}, p={p:.2e}, overlap_days={len(rboth)}")

                if draw_graph:
                    # normalized prices
                    plt.figure(figsize=(9, 3.2))
                    (both / both.iloc[0]).plot(ax=plt.gca())
                    plt.title(f"Normalized Prices: {ti} vs {tj}")
                    plt.xlabel(""); plt.ylabel("Index (=1 at start)")
                    plt.legend(); plt.tight_layout(); plt.show()

                    # scatter of returns
                    plt.figure(figsize=(4.8, 4.4))
                    plt.scatter(rboth[ti], rboth[tj], s=8, alpha=0.6)
                    plt.title(f"Return Scatter: {ti} vs {tj} (rho={rho:+.2f})")
                    plt.xlabel(ti + " returns"); plt.ylabel(tj + " returns")
                    plt.tight_layout(); plt.show()

                edge_stats = {"ti": ti, "tj": tj, "rho": float(rho), "p": float(p), "overlap_days": int(len(rboth))}

        # ---------- 8) Sector–sector mean rho ----------
        sector_matrix = None
        if TICKER_TO_SECTOR:
            rows = []
            inv = {v: k for k, v in node_index_map.items()}
            for u, v, w in adj_list:
                a, b = inv[u], inv[v]
                sa, sb = TICKER_TO_SECTOR.get(a, "?"), TICKER_TO_SECTOR.get(b, "?")
                s1, s2 = (sa, sb) if sa <= sb else (sb, sa)
                rows.append((a, b, w, s1, s2))
            dfE = pd.DataFrame(rows, columns=["ti", "tj", "rho", "sec1", "sec2"])
            sector_matrix = dfE.groupby(["sec1", "sec2"])["rho"].mean().unstack().fillna(0.0)
            print("\nMean edge weight (Spearman rho) by sector pair:")
            print(sector_matrix.round(3))

        # ---------- 9) News/sentiment snapshot ----------
        news_snapshot = None
        has_sent_cols = (
            isinstance(long_df, pd.DataFrame)
            and {"avg_sentiment", "headline_count", "headlines_used"}.issubset(long_df.columns)
        )
        if edge_pair is not None and has_sent_cols:
            ti, tj = edge_pair
            if ti in self.tickers and tj in self.tickers:
                rboth = rets[[ti, tj]].dropna().copy()
                rboth["joint_mag"] = (rboth[ti].abs() + rboth[tj].abs())
                days = rboth.sort_values("joint_mag", ascending=False).head(top_n_news_days).index
                sel = long_df[long_df["Date"].isin(days) & long_df["Ticker"].isin([ti, tj])]
                cols = ["Date", "Ticker", "avg_sentiment", "headline_count", "headlines_used"]
                news_snapshot = sel[cols].sort_values(["Date", "Ticker"])
                print(f"\nNews/Sentiment snapshot for top {top_n_news_days} joint-move days ({ti},{tj}):")
                print(news_snapshot)

        # ---------- 10) Return artifacts ----------
        return {
            "G": G,
            "pos": pos,
            "adj_list": adj_list,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "node_index_map": node_index_map,
            "idx_to_ticker": idx_to_ticker,
            "neighbors": neighbors_out,
            "edge_stats": edge_stats,
            "sector_matrix": sector_matrix,
            "news_snapshot": news_snapshot,
        }
price_df = df.drop_duplicates(subset=["Date", "Ticker"])
corr_graph = StockCorrGraph(price_df)
adj_list, edge_index, edge_weight, node_index_map = corr_graph.build()

# diag = fx.testing(
#     long_df=df,
#     focus_ticker="MMM",
#     edge_pair=("MMM","HON"),
#     draw_graph=True,
# )

C = corr_graph.correlation_matrix()

print(C.shape)