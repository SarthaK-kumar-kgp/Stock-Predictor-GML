import numpy as np
import pandas as pd
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr


class StockCausalityDetector:
    def __init__(self, stock_dataset, alpha=0.05, tau_max=5, verbose=False, min_length=50):
        """
        Initialize the Stock Causality Detector.
        
        Parameters:
        -----------
        stock_dataset : pd.DataFrame
            DataFrame containing 'Ticker' and 'Close' columns
        alpha : float
            Significance level for causal links (default=0.05)
        tau_max : int
            Maximum time lag to test (default=5)
        verbose : bool
            If True, print progress messages (default=False)
        min_length : int
            Minimum required data points (default=50)
        """
        self.alpha = alpha
        self.tau_max = tau_max
        self.verbose = verbose
        self.stock_dataset = stock_dataset
        self.min_length = min_length
        self.results = None
        self.results_df = None

    def detect(self, stock1_prices, stock2_prices, stock1_name="Stock1", stock2_name="Stock2"):
        """
        Detect causality between two stock price series.
        
        Parameters:
        -----------
        stock1_prices : array-like
            Price series for first stock
        stock2_prices : array-like
            Price series for second stock
        stock1_name : str
            Name of first stock
        stock2_name : str
            Name of second stock
            
        Returns:
        --------
        dict : Results dictionary with causality information
        """
        # Clean data: convert to float and drop NaN
        stock1_prices = stock1_prices.astype(float).dropna().reset_index(drop=True)
        stock2_prices = stock2_prices.astype(float).dropna().reset_index(drop=True)
        
        # Align by minimum length
        min_len = min(len(stock1_prices), len(stock2_prices))
        
        # Check if we have enough data
        if min_len < self.min_length:
            if self.verbose:
                print(f"Insufficient data for {stock1_name}-{stock2_name}: {min_len} points (need {self.min_length})")
            return {
                'causality_detected': 0,
                f'{stock1_name}_causes_{stock2_name}': False,
                f'{stock2_name}_causes_{stock1_name}': False,
                'bidirectional': False,
                'details': {},
                'error': f'Insufficient data: {min_len} points'
            }
        
        stock1_prices = stock1_prices.iloc[:min_len].reset_index(drop=True)
        stock2_prices = stock2_prices.iloc[:min_len].reset_index(drop=True)

        return self._detect_stock_causality(
            stock1_prices,
            stock2_prices,
            alpha=self.alpha,
            tau_max=self.tau_max,
            stock1_name=stock1_name,
            stock2_name=stock2_name,
        )

    def _detect_stock_causality(self, stock1_prices, stock2_prices, alpha=0.05, tau_max=5, 
                              stock1_name="Stock1", stock2_name="Stock2"):
        """
        Core causality detection using PCMCI algorithm.
        """
        # Convert to numpy arrays
        stock1_prices = np.array(stock1_prices)
        stock2_prices = np.array(stock2_prices)
        
        # Validate inputs
        if len(stock1_prices) != len(stock2_prices):
            raise ValueError("Both stock price series must have the same length")
        
        if len(stock1_prices) < 10:
            raise ValueError("Need at least 10 time points for reliable causal inference")
        
        # Convert prices to returns (more stationary for causal analysis)
        returns1 = np.diff(np.log(stock1_prices))
        returns2 = np.diff(np.log(stock2_prices))
        
        # Stack into data matrix
        data = np.vstack([returns1, returns2]).T
        var_names = [stock1_name, stock2_name]
        
        # Wrap into Tigramite DataFrame
        dataframe = pp.DataFrame(data, var_names=var_names)
        
        # Initialize independence test
        indep_test = ParCorr(significance='analytic')
        
        # Initialize PCMCI
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=indep_test,
            verbosity=0
        )
        
        # Run PCMCI+
        results = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=alpha)
        
        # Extract causal relationships
        stock1_to_stock2 = False
        stock2_to_stock1 = False
        stock1_to_stock2_strength = 0.0
        stock2_to_stock1_strength = 0.0
        stock1_to_stock2_pval = 1.0
        stock2_to_stock1_pval = 1.0
        stock1_to_stock2_lag = None
        stock2_to_stock1_lag = None
        
        # Check all lags for causal links
        for lag in range(results['p_matrix'].shape[2]):
            # Check Stock1 → Stock2
            pval_1to2 = results['p_matrix'][0, 1, lag]
            val_1to2 = results['val_matrix'][0, 1, lag]
            
            if pval_1to2 < alpha and val_1to2 != 0:
                if abs(val_1to2) > abs(stock1_to_stock2_strength):
                    stock1_to_stock2 = True
                    stock1_to_stock2_strength = val_1to2
                    stock1_to_stock2_pval = pval_1to2
                    stock1_to_stock2_lag = lag
            
            # Check Stock2 → Stock1
            pval_2to1 = results['p_matrix'][1, 0, lag]
            val_2to1 = results['val_matrix'][1, 0, lag]
            
            if pval_2to1 < alpha and val_2to1 != 0:
                if abs(val_2to1) > abs(stock2_to_stock1_strength):
                    stock2_to_stock1 = True
                    stock2_to_stock1_strength = val_2to1
                    stock2_to_stock1_pval = pval_2to1
                    stock2_to_stock1_lag = lag
        
        # Determine if any causality exists
        causality_detected = 1 if (stock1_to_stock2 or stock2_to_stock1) else 0
        bidirectional = stock1_to_stock2 and stock2_to_stock1
        
        # Prepare detailed results
        details = {
            f'{stock1_name}_to_{stock2_name}': {
                'exists': stock1_to_stock2,
                'strength': stock1_to_stock2_strength,
                'p_value': stock1_to_stock2_pval,
                'lag': stock1_to_stock2_lag
            },
            f'{stock2_name}_to_{stock1_name}': {
                'exists': stock2_to_stock1,
                'strength': stock2_to_stock1_strength,
                'p_value': stock2_to_stock1_pval,
                'lag': stock2_to_stock1_lag
            }
        }
        
        return {
            'causality_detected': causality_detected,
            f'{stock1_name}_causes_{stock2_name}': stock1_to_stock2,
            f'{stock2_name}_causes_{stock1_name}': stock2_to_stock1,
            'bidirectional': bidirectional,
            'details': details
        }

    def detect_all_pairs(self):
        """
        Detect causality for all pairs of stocks in the dataset.
        
        Returns:
        --------
        list : List of dictionaries containing results for each pair
        """
        tickers = self.stock_dataset['Ticker'].unique()
        results = []

        total_pairs = len(tickers) * (len(tickers) - 1)
        processed = 0

        for stock1 in tickers:
            for stock2 in tickers:
                if stock1 == stock2:
                    continue
                
                processed += 1
                if self.verbose and processed % 100 == 0:
                    print(f"Processing pair {processed}/{total_pairs}...")
                
                prices1 = self.stock_dataset[self.stock_dataset['Ticker'] == stock1]['Close']
                prices2 = self.stock_dataset[self.stock_dataset['Ticker'] == stock2]['Close']

                # Handle missing or empty series
                if len(prices1) == 0 or len(prices2) == 0:
                    results.append({
                        'stock1': stock1,
                        'stock2': stock2,
                        'causality_detected': 0,
                        'stock1_causes_stock2': False,
                        'stock2_causes_stock1': False,
                        'bidirectional': False,
                        'causality_strength': 0.0,
                        'stock1_to_stock2_strength': 0.0,
                        'stock2_to_stock1_strength': 0.0,
                        'error': 'Empty data'
                    })
                    if self.verbose:
                        print(f"Empty data for {stock1} or {stock2}")
                    continue

                try:
                    result = self.detect(prices1, prices2, stock1_name=stock1, stock2_name=stock2)
                    
                    # Extract strengths from details
                    stock1_to_stock2_strength = 0.0
                    stock2_to_stock1_strength = 0.0
                    
                    if 'details' in result and result['details']:
                        stock1_to_stock2_key = f'{stock1}_to_{stock2}'
                        stock2_to_stock1_key = f'{stock2}_to_{stock1}'
                        
                        if stock1_to_stock2_key in result['details']:
                            stock1_to_stock2_strength = result['details'][stock1_to_stock2_key].get('strength', 0.0)
                        
                        if stock2_to_stock1_key in result['details']:
                            stock2_to_stock1_strength = result['details'][stock2_to_stock1_key].get('strength', 0.0)
                    
                    # Overall causality strength (max of both directions)
                    causality_strength = max(abs(stock1_to_stock2_strength), abs(stock2_to_stock1_strength))
                    
                    results.append({
                        'stock1': stock1,
                        'stock2': stock2,
                        'causality_detected': result['causality_detected'],
                        'stock1_causes_stock2': result.get(f'{stock1}_causes_{stock2}', False),
                        'stock2_causes_stock1': result.get(f'{stock2}_causes_{stock1}', False),
                        'bidirectional': result.get('bidirectional', False),
                        'causality_strength': causality_strength,
                        'stock1_to_stock2_strength': stock1_to_stock2_strength,
                        'stock2_to_stock1_strength': stock2_to_stock1_strength,
                        'error': result.get('error', None)
                    })
                    
                    if self.verbose:
                        print(f"Causality between {stock1} and {stock2}: {result['causality_detected']} (strength: {causality_strength:.4f})")
                        
                except Exception as e:
                    # Return 0 for any error
                    results.append({
                        'stock1': stock1,
                        'stock2': stock2,
                        'causality_detected': 0,
                        'stock1_causes_stock2': False,
                        'stock2_causes_stock1': False,
                        'bidirectional': False,
                        'causality_strength': 0.0,
                        'stock1_to_stock2_strength': 0.0,
                        'stock2_to_stock1_strength': 0.0,
                        'error': str(e)
                    })
                    if self.verbose:
                        print(f"Failed for {stock1} -> {stock2}: {e}")
        
        self.results = results
        self.results_df = pd.DataFrame(results)
        return results

    def get_binary_matrix(self):
        """
        Get binary causality matrix (0/1).
        
        Returns:
        --------
        pd.DataFrame : n×n binary matrix where 1 indicates causality exists
        """
        if self.results_df is None:
            raise ValueError("No results available. Run detect_all_pairs() first.")
        
        # Get unique list of all stocks
        all_stocks = sorted(set(self.results_df['stock1'].unique()) | set(self.results_df['stock2'].unique()))
        n = len(all_stocks)
        
        # Create index mapping
        stock_to_idx = {stock: idx for idx, stock in enumerate(all_stocks)}
        
        # Initialize matrix
        binary_matrix = np.zeros((n, n))
        
        # Fill the matrix
        for _, row in self.results_df.iterrows():
            stock1 = row['stock1']
            stock2 = row['stock2']
            
            i = stock_to_idx[stock1]
            j = stock_to_idx[stock2]
            
            # Binary matrix: 1 if causality detected, 0 otherwise
            binary_matrix[i, j] = row['causality_detected']
        
        # Convert to DataFrame
        binary_df = pd.DataFrame(binary_matrix, index=all_stocks, columns=all_stocks)
        
        if self.verbose:
            print(f"\nBinary Matrix Statistics:")
            print(f"Total causal relationships: {binary_matrix.sum()}")
            print(f"Percentage with causality: {(binary_matrix.sum() / (n * (n-1))) * 100:.2f}%")
        
        return binary_df

    def get_strength_matrix(self):
        """
        Get causality strength matrix.
        
        Returns:
        --------
        pd.DataFrame : n×n strength matrix with directional causality strengths
        """
        if self.results_df is None:
            raise ValueError("No results available. Run detect_all_pairs() first.")
        
        # Get unique list of all stocks
        all_stocks = sorted(set(self.results_df['stock1'].unique()) | set(self.results_df['stock2'].unique()))
        n = len(all_stocks)
        
        # Create index mapping
        stock_to_idx = {stock: idx for idx, stock in enumerate(all_stocks)}
        
        # Initialize matrix
        strength_matrix = np.zeros((n, n))
        
        # Fill the matrix
        for _, row in self.results_df.iterrows():
            stock1 = row['stock1']
            stock2 = row['stock2']
            
            i = stock_to_idx[stock1]
            j = stock_to_idx[stock2]
            
            # Strength matrix: use stock1_to_stock2_strength for directional strength
            strength_matrix[i, j] = row['stock1_to_stock2_strength']
        
        # Convert to DataFrame
        strength_df = pd.DataFrame(strength_matrix, index=all_stocks, columns=all_stocks)
        
        if self.verbose:
            print(f"\nStrength Matrix Statistics:")
            print(f"Average strength (non-zero): {strength_matrix[strength_matrix != 0].mean():.4f}")
            print(f"Max strength: {strength_matrix.max():.4f}")
            print(f"Min strength: {strength_matrix.min():.4f}")
        
        return strength_df

##How to use the functions above:

# detector = StockCausalityDetector(
#     df_test,
#     alpha=0.05, 
#     tau_max=5, 
#     verbose=True, 
#     min_length=10
# )


# results = detector.detect_all_pairs()

# # Get results as DataFrame
# results_df = detector.results_df
# results_df


# # Get binary matrix (0/1)
# binary_matrix = detector.get_binary_matrix()
# print("\n=== Binary Causality Matrix ===")
# binary_matrix


# # Get strength matrix
# strength_matrix = detector.get_strength_matrix()
# print("\n=== Causality Strength Matrix ===")
# strength_matrix