# Implementation Guide
## How to Use the Institutional Backtest Code

---

## Quick Start

### 1. Environment Setup

```bash
# Install required packages
pip install yfinance pandas numpy matplotlib seaborn --break-system-packages

# Or use requirements.txt
pip install -r requirements.txt --break-system-packages
```

### 2. Basic Usage

```bash
# Run complete backtest
python run_backtest.py

# This will:
# - Load data
# - Generate signals
# - Run portfolio simulation
# - Perform robustness tests
# - Create visualizations
# - Generate reports
```

### 3. Output Files

All results saved to `/mnt/user-data/outputs/`:

```
equity_curve.png              # Portfolio value over time
rolling_sharpe.png            # Rolling 1-year Sharpe ratio
return_distribution.png       # Histogram of trade returns
monthly_returns.png           # Heatmap of monthly performance
monte_carlo.png              # Monte Carlo simulation results
regime_performance.png        # Performance by market regime
parameter_sensitivity.png     # Parameter robustness test

trade_log.csv                # Complete trade history
equity_curve.csv             # Daily portfolio values
performance_metrics.csv       # Summary statistics
is_oos_comparison.csv        # In-sample vs out-of-sample
regime_analysis.csv          # Regime-segmented results
SUMMARY_REPORT.txt           # Executive summary
```

---

## Code Architecture

### Module Overview

```
institutional_backtest.py     # Core backtest engine
├── StrategyConfig            # Configuration parameters
├── DataLoader                # Data acquisition & cleaning
├── SignalGenerator           # Signal generation logic
├── PortfolioSimulator        # Full portfolio simulation
└── PerformanceAnalyzer       # Metrics calculation

robustness_testing.py         # Validation framework
├── RobustnessTester          # IS/OOS, parameter sensitivity
├── RegimeAnalyzer            # Market regime classification
└── StrategyComparison        # Original vs rebuilt comparison

visualization.py              # Charting & plotting
└── StrategyVisualizer        # All visualization functions

run_backtest.py              # Main orchestration script
```

---

## Configuration

### Modifying Parameters

Edit `institutional_backtest.py` → `StrategyConfig` class:

```python
class StrategyConfig:
    # Data Parameters
    START_DATE = "2020-01-01"      # Backtest start
    END_DATE = "2026-02-09"        # Backtest end
    
    # Universe Parameters
    UNIVERSE_SIZE = 500            # Top N stocks
    MIN_PRICE = 10                 # Minimum stock price (₹)
    MIN_ADV = 5_000_000           # Min avg daily value (₹)
    
    # Signal Parameters
    BREAKOUT_THRESHOLD = 0.05      # 5% breakout
    VOLUME_MULTIPLIER = 1.0        # Volume expansion factor
    EMA_PERIOD = 200               # Trend filter period
    
    # Execution
    HOLDING_PERIOD = 20            # Days to hold
    COOLDOWN_PERIOD = 63           # Days before retrade (3 months)
    
    # Transaction Costs (basis points)
    BROKERAGE_BPS = 3             # 0.03%
    STT_BPS = 10                  # 0.10% (sell only)
    EXCHANGE_BPS = 3.5            # 0.035%
    SLIPPAGE_BPS = 12.5           # 0.125% per side
    
    # Portfolio
    INITIAL_CAPITAL = 10_00_000   # ₹10 lakhs
    MAX_POSITIONS = 20            # Max concurrent positions
    MAX_POSITION_SIZE = 0.10      # 10% max per position
    
    # Risk Management
    MAX_DRAWDOWN_STOP = 0.20      # 20% kill switch
    USE_REGIME_FILTER = True      # Market trend filter
```

---

## Usage Examples

### Example 1: Run with Custom Parameters

```python
from institutional_backtest import StrategyConfig, DataLoader, SignalGenerator, PortfolioSimulator

# Create custom configuration
config = StrategyConfig()
config.BREAKOUT_THRESHOLD = 0.06  # Test 6% breakout
config.HOLDING_PERIOD = 10         # Shorter holding period
config.MAX_POSITIONS = 15          # Fewer positions

# Run backtest
loader = DataLoader(config)
price_data = loader.download_price_data(tickers)

signal_gen = SignalGenerator(config)
signals = {ticker: signal_gen.generate_signals(
    signal_gen.compute_indicators(df)
) for ticker, df in price_data.items()}

simulator = PortfolioSimulator(config, price_data)
equity_curve = simulator.run_backtest(signals)
```

### Example 2: Parameter Sweep

```python
from robustness_testing import RobustnessTester

tester = RobustnessTester(config)

# Test multiple parameter combinations
sensitivity_results = tester.parameter_sensitivity(
    price_data=price_data,
    signal_generator_class=SignalGenerator,
    simulator_class=PortfolioSimulator
)

print(sensitivity_results)
```

### Example 3: Monte Carlo Analysis

```python
from robustness_testing import RobustnessTester

# After running backtest
tester = RobustnessTester(config)
mc_results = tester.monte_carlo_resampling(
    trade_log=simulator.trade_log,
    n_simulations=1000,
    initial_capital=config.INITIAL_CAPITAL
)

print(f"Original CAGR: {mc_results['original_cagr']:.2%}")
print(f"Mean Simulated: {mc_results['mean_cagr']:.2%}")
print(f"95% CI: {mc_results['cagr_confidence_95']}")
```

### Example 4: Regime Analysis

```python
from robustness_testing import RegimeAnalyzer

analyzer = RegimeAnalyzer(index_ticker="^NSEI")
index_data = analyzer.load_index_data(config.START_DATE, config.END_DATE)

regime_performance = analyzer.analyze_by_regime(
    trade_log=simulator.trade_log,
    equity_curve=equity_curve
)

print(regime_performance)
```

### Example 5: Custom Visualization

```python
from visualization import StrategyVisualizer

viz = StrategyVisualizer()

# Equity curve
viz.plot_equity_curve(
    equity_curve,
    title="My Custom Strategy",
    save_path="/mnt/user-data/outputs/my_equity.png"
)

# Return distribution
viz.plot_return_distribution(
    simulator.trade_log,
    save_path="/mnt/user-data/outputs/my_returns.png"
)

# Monthly heatmap
viz.plot_monthly_returns_heatmap(
    equity_curve,
    save_path="/mnt/user-data/outputs/my_monthly.png"
)
```

---

## Advanced Features

### 1. In-Sample / Out-of-Sample Testing

```python
from robustness_testing import RobustnessTester

tester = RobustnessTester(config)

# Split data
in_sample, out_of_sample = tester.train_test_split(
    start_date=config.START_DATE,
    end_date=config.END_DATE,
    split_ratio=0.70
)

# Run IS backtest
sim_is = PortfolioSimulator(config, price_data)
equity_is = sim_is.run_backtest(signals, in_sample[0], in_sample[1])

# Run OOS backtest
sim_oos = PortfolioSimulator(config, price_data)
equity_oos = sim_oos.run_backtest(signals, out_of_sample[0], out_of_sample[1])

# Compare
from institutional_backtest import PerformanceAnalyzer
analyzer = PerformanceAnalyzer()
metrics_is = analyzer.calculate_metrics(equity_is, sim_is.trade_log)
metrics_oos = analyzer.calculate_metrics(equity_oos, sim_oos.trade_log)

print(f"IS Sharpe: {metrics_is['sharpe_ratio']:.2f}")
print(f"OOS Sharpe: {metrics_oos['sharpe_ratio']:.2f}")
```

### 2. Add Custom Filters

```python
class CustomSignalGenerator(SignalGenerator):
    """Extended signal generator with custom filters"""
    
    def generate_signals(self, df, breakout_threshold=None):
        # Call parent method
        df = super().generate_signals(df, breakout_threshold)
        
        # Add custom filter: RSI must be < 70
        df['rsi'] = self.calculate_rsi(df['Close'], period=14)
        df['signal'] = df['signal'] & (df['rsi'] < 70)
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

# Use custom generator
signal_gen = CustomSignalGenerator(config)
```

### 3. Alternative Position Sizing

```python
class VolatilityScaledSimulator(PortfolioSimulator):
    """Position sizing based on volatility"""
    
    def calculate_position_size(self, ticker, price):
        # Get historical volatility
        df = self.price_data[ticker]
        volatility = df['Close'].pct_change().rolling(30).std().iloc[-1]
        
        # Target volatility per position
        target_vol = 0.02  # 2% per position
        
        # Scale position size inversely with volatility
        base_allocation = self.capital / self.config.MAX_POSITIONS
        vol_adjusted = base_allocation * (target_vol / volatility)
        
        # Apply limits
        max_position = self.capital * self.config.MAX_POSITION_SIZE
        allocation = min(vol_adjusted, max_position)
        
        shares = int(allocation / price)
        return shares

# Use volatility-scaled sizing
simulator = VolatilityScaledSimulator(config, price_data)
```

---

## Troubleshooting

### Common Issues

**1. "ModuleNotFoundError: No module named 'yfinance'"**
```bash
pip install yfinance --break-system-packages
```

**2. "NameError: name 'plt' is not defined"**
```bash
pip install matplotlib --break-system-packages
```

**3. "Empty DataFrame" or "No data returned"**
- Check ticker format (should be "RELIANCE.NS" for NSE)
- Verify date range is valid
- Ensure internet connection for yfinance

**4. "Backtest runs but no trades"**
- Signals may be too restrictive
- Try lowering `BREAKOUT_THRESHOLD` to 0.04
- Check if liquidity filters are too strict

**5. "Memory Error"**
- Reduce `UNIVERSE_SIZE`
- Shorten `START_DATE` to `END_DATE` range
- Use fewer stocks in initial testing

---

## Performance Optimization

### For Faster Backtests

```python
# 1. Reduce universe size for testing
config.UNIVERSE_SIZE = 50  # Instead of 500

# 2. Use shorter date range
config.START_DATE = "2024-01-01"  # Instead of 2020

# 3. Skip Monte Carlo in initial runs
# Comment out MC section in run_backtest.py

# 4. Use multiprocessing for signal generation
from multiprocessing import Pool

def generate_signals_parallel(ticker_df_tuple):
    ticker, df = ticker_df_tuple
    signal_gen = SignalGenerator(config)
    return ticker, signal_gen.generate_signals(
        signal_gen.compute_indicators(df)
    )

with Pool(processes=4) as pool:
    results = pool.map(generate_signals_parallel, price_data.items())
    signals_dict = dict(results)
```

---

## Data Sources

### Recommended Alternatives to yfinance

For production use:

1. **NSE Data API** (nsetools)
   ```python
   from nsetools import Nse
   nse = Nse()
   ```

2. **Quandl**
   ```python
   import quandl
   df = quandl.get("NSE/RELIANCE")
   ```

3. **Alpha Vantage**
   ```python
   from alpha_vantage.timeseries import TimeSeries
   ts = TimeSeries(key='YOUR_API_KEY')
   ```

4. **Bloomberg Terminal** (institutional)
5. **Refinitiv Eikon** (institutional)

---

## Best Practices

### 1. Version Control
```bash
git init
git add *.py
git commit -m "Initial institutional backtest implementation"
```

### 2. Configuration Management
```python
# Use separate config files for different strategies
import json

with open('config.json', 'r') as f:
    params = json.load(f)
    
config = StrategyConfig()
config.BREAKOUT_THRESHOLD = params['breakout']
config.HOLDING_PERIOD = params['holding']
```

### 3. Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Starting backtest...")
```

### 4. Unit Testing
```python
import unittest

class TestSignalGenerator(unittest.TestCase):
    def test_breakout_detection(self):
        # Create mock data
        df = pd.DataFrame({
            'Close': [100, 100, 106],  # 6% breakout on day 2
            'Volume': [1000, 1000, 2000],
        })
        
        signal_gen = SignalGenerator(config)
        df = signal_gen.compute_indicators(df)
        df = signal_gen.generate_signals(df)
        
        self.assertTrue(df['signal'].iloc[-1])

if __name__ == '__main__':
    unittest.main()
```

---

## Next Steps

### Immediate Actions
1. Run `python run_backtest.py` to see framework in action
2. Review output files in `/mnt/user-data/outputs/`
3. Modify parameters in `StrategyConfig`
4. Test on different date ranges

### Research Extensions
1. Add stop-loss mechanism
2. Implement volatility-based position sizing
3. Test alternative trend filters (50 DMA, MACD)
4. Add fundamental filters (P/E, debt ratios)
5. Create ensemble with mean reversion strategy

### Production Deployment
1. Connect to real-time data feed
2. Implement order execution system
3. Add monitoring & alerting
4. Set up daily reconciliation
5. Create risk dashboard

---

## Support & Resources

### Documentation
- Code comments in each module
- Docstrings for all classes and methods
- Type hints for function parameters

### Examples
- See `run_backtest.py` for full workflow
- Check function docstrings for usage

### Community
- GitHub Issues for bug reports
- Discussions for strategy ideas

---

**Last Updated:** February 9, 2026  
**Version:** 1.0.0  
**License:** MIT

---

*This implementation represents institutional best practices for strategy backtesting. Always validate results with paper trading before live deployment.*
