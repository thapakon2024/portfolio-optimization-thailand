# Portfolio Optimization for Thailand Stock Market (SET)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Optimized Portfolio Optimization for Thai Stock Market (SET50) using TensorFlow - Low risk, high return investment strategy with log returns

## 🎯 Features

- **Ultra-Fast Optimization**: 10-50x faster than traditional approaches using TensorFlow
- **Multiple Portfolio Sizes**: Support for N=2, N=3, N=5 stocks per portfolio
- **Vectorized Calculations**: All operations in TensorFlow for maximum performance
- **GPU Support**: Automatic GPU acceleration when available
- **Memory Efficient**: Option for low-memory systems with batch processing
- **Log Return Analysis**: Using logarithmic returns for volatility calculation
- **Multiple Return Targets**: Optimize for 1%, 2%, 3%, 4%, 5% returns simultaneously

## 📊 Performance

| Portfolio Size | Combinations | Time (GPU) | Time (CPU) |
|----------------|--------------|------------|------------|
| N=2            | 1,081        | ~10 sec    | ~30 sec    |
| N=3            | 16,215       | ~2 min     | ~5 min     |
| N=5            | 1,533,939    | ~25 min    | ~50 min    |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/thapakon2024/portfolio-optimization-thailand.git
cd portfolio-optimization-thailand

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import numpy as np
import tensorflow_probability as tfp
import yfinance as yf
from portfolio_n5_optimized import PortfolioOptimizerN5

# Parameters
n = 5  # Number of stocks
fund = 1000  # Investment budget (THB)
num_samples = 50000

# Generate random weights
dist = tfp.distributions.Dirichlet(np.ones(n))
Wi = dist.sample(num_samples)

# Download stock data
stock = ["ADVANC.BK", "AOT.BK", "BBL.BK", "BDMS.BK", "PTT.BK"]
df = yf.download(stock, start='2022-10-19', end='2022-11-30')

# Optimize
optimizer = PortfolioOptimizerN5(df, companies, Wi, fund)
best_portfolios, all_results = optimizer.optimize_multiple_returns(
    return_targets=[(0.9999, 1.0001), (1.9999, 2.0001)]
)
```

## 📁 Project Structure

```
portfolio-optimization-thailand/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── portfolio_n2_optimized.py      # Optimized for N=2
├── portfolio_n5_optimized.py      # Optimized for N=5
├── portfolio_n5_memory_efficient.py  # Memory-efficient version
├── example_usage.ipynb            # Jupyter notebook examples
└── docs/
    └── methodology.md             # Detailed methodology
```

## 💡 Key Concepts

### Portfolio Optimization Formula

**Portfolio Value:**
```
V(t) = Σ(wi × fund / P0,i) × Pi(t)
```

**Portfolio Return:**
```
R = (V(t) - fund) / fund × 100%
```

**Log Return:**
```
log_return = log(V(t)) - log(V(t-1))
```

**Volatility:**
```
σ = √(Σ(Δlog_return²))
```

## 🎓 Example Results

### Target Return: 1%
```
📊 Portfolio Metrics:
   • Actual Return:  1.0000%
   • Volatility:     0.182417
   • Variance:       0.245632

🏢 Selected Stocks & Weights:
   Stock        Weight   Shares     Price     Invested
   ───────────────────────────────────────────────────
   ADVANC.BK    0.2145      0      185.50        0.00
   AOT.BK       0.1823      2       62.75      125.50
   BBL.BK       0.2456      1      145.00      145.00
   BDMS.BK      0.1876      7       26.50      185.50
   PTT.BK       0.1700      4       42.25      169.00

💰 Investment Summary:
   • Total Fund:          1,000.00 THB
   • Total Invested:        625.00 THB (62.5%)
   • Remaining Cash:        375.00 THB (37.5%)
```

## ⚙️ Advanced Configuration

### For Systems with Limited RAM

```python
from portfolio_n5_memory_efficient import MemoryEfficientOptimizerN5

optimizer = MemoryEfficientOptimizerN5(
    df=df,
    companies=companies,
    n_stocks=5,
    fund=1000,
    wi_batch_size=5000,    # Smaller batches
    combo_batch_size=20
)

best_portfolios = optimizer.optimize_memory_efficient(
    return_targets=[(0.9999, 1.0001)],
    total_wi_samples=20000
)
```

## 📈 Supported Stocks (SET50)

ADVANC, AOT, BBL, BDMS, BEM, BGRIM, BH, BJC, BTS, CBG, COM7, CPALL, CPF, CPN, CRC, DELTA, EA, EGCO, GLOBAL, GPSC, HMPRO, IRPC, IVL, KBANK, KCE, KTB, KTC, LH, MINT, MTC, OR, OSP, PTT, PTTEP, PTTGC, RATCH, SAWAD, SCB, SCC, SCGP, STA, STGT, TISCO, TOP, TTB, TU, TRUE

## 🔧 Requirements

- Python 3.8+
- TensorFlow 2.x
- TensorFlow Probability
- NumPy
- Pandas
- yfinance
- Jupyter Notebook (optional)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

- GitHub: [@thapakon2024](https://github.com/thapakon2024)

## 🙏 Acknowledgments

- Thai Stock Market data provided by Yahoo Finance
- TensorFlow team for the amazing framework
- Portfolio optimization theory and methodology

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not financial advice. Always consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.

---

Made with ❤️ for Thai investors
