# SIAANet — Graph-Based Stock Market Forecasting

Predicts **7-day ahead Close Returns (CR)** for S&P 500 stocks using a **Graph Transformer** that models both temporal and structural relationships.

---

## Features

- Multi-edge graph:
  - Correlation-based edges  
  - Industry similarity edges  
  - Causal edges (PCMCI)  
  - Embedding similarity edges
- Node embeddings initialized with **sinusoidal positional encodings**  
- 1D Conv-based temporal node encoder  
- Learnable edge embeddings with gated attention  
- 7 separate prediction heads for multi-step forecasting  
- Multi-stock and multi-horizon forecasting  

---

## Data

- **OHLCV data:** Yahoo Finance  
- **VIX volatility index:** Yahoo Finance  
- **Industry and sector metadata:** Yahoo Finance  
- **Preprocessing:**
  - Rolling averages, log returns, volatility, volume features  
  - Standardized numeric features (Z-score)  
  - Cyclic encodings: day, month, quarter  

---

## Model Architecture

- Graph Transformer (4 layers, 8 attention heads, 128 hidden dim)  
- Edge-gated attention  
- Node encoder: 1D Convolutions over sliding 30-day window  
- Prediction: 7 independent heads, one per future day  
- Loss: MSE across all stocks and horizons  

**Architecture Diagram Placeholder**  
`![Architecture Diagram](./images/architecture.png)`

---

## Results

- **Directional accuracy (UP/DOWN):** ~70% for 1–3 day horizons  
- Chronos baseline (pretrained time-series model): ~48%  
- Regression performance: MAPE ≤10% for “good” stocks  
- Identified 19 consistently predictable stocks  

---


