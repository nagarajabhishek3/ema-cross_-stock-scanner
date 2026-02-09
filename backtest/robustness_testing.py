"""
Robustness Testing & Regime Analysis Module
============================================

This module provides:
- In-sample / Out-of-sample testing
- Parameter sensitivity analysis
- Monte Carlo simulation
- Regime-based performance analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class RobustnessTester:
    """Perform robustness testing on strategy"""
    
    def __init__(self, config):
        self.config = config
    
    def train_test_split(
        self,
        start_date: str,
        end_date: str,
        split_ratio: float = 0.70
    ) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        """
        Split data into in-sample and out-of-sample periods.
        Uses chronological split.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            split_ratio: Fraction for in-sample (default 0.70 for 70/30)
        
        Returns:
            ((is_start, is_end), (oos_start, oos_end))
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        total_days = (end - start).days
        split_days = int(total_days * split_ratio)
        
        is_end = start + pd.Timedelta(days=split_days)
        oos_start = is_end + pd.Timedelta(days=1)
        
        in_sample = (start.strftime('%Y-%m-%d'), is_end.strftime('%Y-%m-%d'))
        out_of_sample = (oos_start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        
        return in_sample, out_of_sample
    
    def parameter_sensitivity(
        self,
        price_data: Dict,
        signal_generator_class,
        simulator_class
    ) -> pd.DataFrame:
        """
        Test strategy across different parameter combinations.
        
        Parameters tested:
        - Breakout levels: 4%, 5%, 6%
        - Holding periods: 5, 10, 20 days
        """
        
        breakout_levels = [0.04, 0.05, 0.06]
        holding_periods = [5, 10, 20]
        
        results = []
        
        print("\n" + "="*60)
        print("PARAMETER SENSITIVITY ANALYSIS")
        print("="*60 + "\n")
        
        for breakout in breakout_levels:
            for holding in holding_periods:
                
                print(f"Testing: Breakout={breakout:.1%}, Holding={holding}d")
                
                # Create modified config
                config_mod = type(self.config)()
                config_mod.BREAKOUT_THRESHOLD = breakout
                config_mod.HOLDING_PERIOD = holding
                
                # Generate signals with new parameters
                signal_gen = signal_generator_class(config_mod)
                signals_dict = {}
                
                for ticker, df in price_data.items():
                    df_ind = signal_gen.compute_indicators(df)
                    df_sig = signal_gen.generate_signals(df_ind, breakout)
                    signals_dict[ticker] = df_sig
                
                # Run backtest
                sim = simulator_class(config_mod, price_data)
                equity = sim.run_backtest(signals_dict)
                
                # Calculate metrics
                from institutional_backtest import PerformanceAnalyzer
                analyzer = PerformanceAnalyzer()
                metrics = analyzer.calculate_metrics(equity, sim.trade_log)
                
                # Store results
                results.append({
                    'breakout_pct': breakout,
                    'holding_days': holding,
                    'cagr': metrics['cagr'],
                    'sharpe': metrics['sharpe_ratio'],
                    'max_dd': metrics['max_drawdown'],
                    'win_rate': metrics['win_rate'],
                    'total_trades': metrics['total_trades'],
                    'avg_return_net': metrics['avg_return_net']
                })
        
        results_df = pd.DataFrame(results)
        
        print("\nParameter Sensitivity Results:")
        print(results_df.to_string(index=False))
        
        return results_df
    
    def monte_carlo_resampling(
        self,
        trade_log: List[Dict],
        n_simulations: int = 1000,
        initial_capital: float = 1000000
    ) -> Dict:
        """
        Perform Monte Carlo simulation by resampling trade sequence.
        
        This tests strategy robustness by randomizing trade order
        to see if performance was due to lucky timing.
        
        Args:
            trade_log: List of completed trades
            n_simulations: Number of random sequences to test
            initial_capital: Starting capital
        
        Returns:
            Dictionary with simulation results
        """
        
        if not trade_log:
            return {'error': 'No trades to simulate'}
        
        print("\n" + "="*60)
        print(f"MONTE CARLO SIMULATION ({n_simulations} iterations)")
        print("="*60 + "\n")
        
        trades_df = pd.DataFrame(trade_log)
        
        # Original performance
        original_returns = trades_df['return_net'].values
        original_cagr = self._calculate_cagr_from_returns(
            original_returns, initial_capital, len(trades_df)
        )
        
        # Run simulations
        simulation_cagrs = []
        simulation_sharpes = []
        simulation_max_dds = []
        
        np.random.seed(42)  # For reproducibility
        
        for i in range(n_simulations):
            # Resample with replacement
            shuffled_returns = np.random.choice(
                original_returns, 
                size=len(original_returns), 
                replace=True
            )
            
            # Calculate equity curve
            equity = [initial_capital]
            for ret in shuffled_returns:
                new_equity = equity[-1] * (1 + ret)
                equity.append(new_equity)
            
            equity = np.array(equity)
            
            # Calculate metrics
            cagr = self._calculate_cagr_from_equity(equity, len(trades_df))
            sharpe = self._calculate_sharpe_from_returns(shuffled_returns)
            max_dd = self._calculate_max_drawdown(equity)
            
            simulation_cagrs.append(cagr)
            simulation_sharpes.append(sharpe)
            simulation_max_dds.append(max_dd)
        
        simulation_cagrs = np.array(simulation_cagrs)
        simulation_sharpes = np.array(simulation_sharpes)
        simulation_max_dds = np.array(simulation_max_dds)
        
        # Calculate percentiles
        percentiles = [5, 25, 50, 75, 95]
        
        results = {
            'original_cagr': original_cagr,
            'mean_cagr': simulation_cagrs.mean(),
            'median_cagr': np.median(simulation_cagrs),
            'cagr_percentiles': {p: np.percentile(simulation_cagrs, p) for p in percentiles},
            'cagr_confidence_95': (
                np.percentile(simulation_cagrs, 2.5),
                np.percentile(simulation_cagrs, 97.5)
            ),
            'pct_better_than_original': (simulation_cagrs < original_cagr).mean(),
            'sharpe_distribution': simulation_sharpes,
            'max_dd_distribution': simulation_max_dds
        }
        
        print(f"Original CAGR:          {original_cagr:.2%}")
        print(f"Mean Simulated CAGR:    {results['mean_cagr']:.2%}")
        print(f"Median Simulated CAGR:  {results['median_cagr']:.2%}")
        print(f"\nCAGR Percentiles:")
        for p, v in results['cagr_percentiles'].items():
            print(f"  {p}th percentile: {v:.2%}")
        print(f"\n95% Confidence Interval: ({results['cagr_confidence_95'][0]:.2%}, "
              f"{results['cagr_confidence_95'][1]:.2%})")
        print(f"\nProbability of beating original: {1-results['pct_better_than_original']:.1%}")
        
        return results
    
    def _calculate_cagr_from_returns(
        self, 
        returns: np.ndarray, 
        initial_capital: float,
        n_days: int
    ) -> float:
        """Calculate CAGR from return series"""
        cumulative_return = np.prod(1 + returns) - 1
        final_value = initial_capital * (1 + cumulative_return)
        years = n_days / 252  # Assuming 252 trading days
        cagr = (final_value / initial_capital) ** (1 / years) - 1
        return cagr
    
    def _calculate_cagr_from_equity(self, equity: np.ndarray, n_days: int) -> float:
        """Calculate CAGR from equity curve"""
        years = n_days / 252
        cagr = (equity[-1] / equity[0]) ** (1 / years) - 1
        return cagr
    
    def _calculate_sharpe_from_returns(self, returns: np.ndarray, rf: float = 0.07) -> float:
        """Calculate Sharpe ratio from returns"""
        rf_per_trade = rf / 252
        excess_returns = returns - rf_per_trade
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return sharpe
    
    def _calculate_max_drawdown(self, equity: np.ndarray) -> float:
        """Calculate maximum drawdown from equity curve"""
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax
        return drawdown.min()


class RegimeAnalyzer:
    """Analyze strategy performance across different market regimes"""
    
    def __init__(self, index_ticker: str = "^NSEI"):
        self.index_ticker = index_ticker
        self.index_data = None
    
    def load_index_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load index data for regime classification"""
        
        print(f"\nLoading index data ({self.index_ticker})...")
        
        import yfinance as yf
        
        df = yf.download(
            self.index_ticker,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        if df.empty:
            print("Warning: Could not load index data")
            return None
        
        # Calculate metrics for regime classification
        df['return_6m'] = df['Close'].pct_change(126)  # 6 months ≈ 126 trading days
        df['volatility_30d'] = df['Close'].pct_change().rolling(30).std() * np.sqrt(252)
        
        self.index_data = df
        return df
    
    def classify_regime(
        self,
        date: pd.Timestamp,
        bull_threshold: float = 0.15,
        bear_threshold: float = -0.10,
        high_vol_threshold: float = 0.25
    ) -> str:
        """
        Classify market regime for a given date.
        
        Regimes:
        - Bull: 6-month return > 15%
        - Bear: 6-month return < -10%
        - High Vol: 30-day annualized vol > 25%
        - Low Vol: 30-day annualized vol < 25%
        - Sideways: Everything else
        """
        
        if self.index_data is None or date not in self.index_data.index:
            return 'unknown'
        
        row = self.index_data.loc[date]
        
        ret_6m = row.get('return_6m', 0)
        vol_30d = row.get('volatility_30d', 0)
        
        # Trend classification
        if ret_6m > bull_threshold:
            trend = 'bull'
        elif ret_6m < bear_threshold:
            trend = 'bear'
        else:
            trend = 'sideways'
        
        # Volatility classification
        vol_regime = 'high_vol' if vol_30d > high_vol_threshold else 'low_vol'
        
        # Combined regime
        regime = f"{trend}_{vol_regime}"
        
        return regime
    
    def analyze_by_regime(
        self,
        trade_log: List[Dict],
        equity_curve: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Segment performance by market regime.
        
        Returns DataFrame with metrics for each regime.
        """
        
        if self.index_data is None:
            print("Error: Index data not loaded")
            return pd.DataFrame()
        
        print("\n" + "="*60)
        print("REGIME ANALYSIS")
        print("="*60 + "\n")
        
        # Classify each trade
        trades_df = pd.DataFrame(trade_log)
        
        if trades_df.empty:
            print("No trades to analyze")
            return pd.DataFrame()
        
        trades_df['regime'] = trades_df['entry_date'].apply(self.classify_regime)
        
        # Calculate metrics by regime
        regime_metrics = []
        
        for regime in trades_df['regime'].unique():
            regime_trades = trades_df[trades_df['regime'] == regime]
            
            n_trades = len(regime_trades)
            win_rate = (regime_trades['return_net'] > 0).mean()
            avg_return = regime_trades['return_net'].mean()
            median_return = regime_trades['return_net'].median()
            
            regime_metrics.append({
                'regime': regime,
                'n_trades': n_trades,
                'win_rate': win_rate,
                'avg_return': avg_return,
                'median_return': median_return,
                'total_pnl': regime_trades['pnl_net'].sum()
            })
        
        regime_df = pd.DataFrame(regime_metrics)
        regime_df = regime_df.sort_values('n_trades', ascending=False)
        
        print("\nPerformance by Regime:")
        print(regime_df.to_string(index=False))
        
        return regime_df


class StrategyComparison:
    """Compare original vs rebuilt strategy"""
    
    @staticmethod
    def create_comparison_table(
        original_metrics: Dict,
        rebuilt_metrics: Dict
    ) -> pd.DataFrame:
        """Create side-by-side comparison table"""
        
        comparison = pd.DataFrame({
            'Metric': [
                'CAGR',
                'Sharpe Ratio',
                'Sortino Ratio',
                'Max Drawdown',
                'Win Rate',
                'Total Trades',
                'Avg Return (Net)',
                'Annual Turnover'
            ],
            'Original': [
                f"{original_metrics.get('cagr', 0):.2%}",
                f"{original_metrics.get('sharpe_ratio', 0):.2f}",
                f"{original_metrics.get('sortino_ratio', 0):.2f}",
                f"{original_metrics.get('max_drawdown', 0):.2%}",
                f"{original_metrics.get('win_rate', 0):.2%}",
                f"{original_metrics.get('total_trades', 0):.0f}",
                f"{original_metrics.get('avg_return_net', 0):.2%}",
                f"{original_metrics.get('turnover_annual', 0):.1f}x"
            ],
            'Rebuilt (Institutional)': [
                f"{rebuilt_metrics.get('cagr', 0):.2%}",
                f"{rebuilt_metrics.get('sharpe_ratio', 0):.2f}",
                f"{rebuilt_metrics.get('sortino_ratio', 0):.2f}",
                f"{rebuilt_metrics.get('max_drawdown', 0):.2%}",
                f"{rebuilt_metrics.get('win_rate', 0):.2%}",
                f"{rebuilt_metrics.get('total_trades', 0):.0f}",
                f"{rebuilt_metrics.get('avg_return_net', 0):.2%}",
                f"{rebuilt_metrics.get('turnover_annual', 0):.1f}x"
            ]
        })
        
        return comparison
    
    @staticmethod
    def generate_verdict(
        metrics: Dict,
        monte_carlo_results: Dict = None
    ) -> str:
        """
        Generate institutional verdict on strategy deployability.
        
        Considers:
        - Risk-adjusted returns
        - Stability
        - Transaction cost impact
        - Capital scalability
        """
        
        verdict_parts = []
        
        verdict_parts.append("="*60)
        verdict_parts.append("INSTITUTIONAL VERDICT")
        verdict_parts.append("="*60)
        
        # Overall assessment
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown', 0)
        win_rate = metrics.get('win_rate', 0)
        
        verdict_parts.append("\n1. DEPLOYABILITY ASSESSMENT:")
        
        if sharpe > 1.5 and max_dd > -0.25:
            verdict_parts.append("   ✓ DEPLOYABLE - Strong risk-adjusted returns")
        elif sharpe > 1.0 and max_dd > -0.30:
            verdict_parts.append("   ⚠ CONDITIONALLY DEPLOYABLE - Acceptable with limits")
        else:
            verdict_parts.append("   ✗ NOT DEPLOYABLE - Insufficient risk-adjusted returns")
        
        # Capital size recommendation
        verdict_parts.append("\n2. RECOMMENDED CAPITAL SIZE:")
        
        avg_positions = metrics.get('avg_positions', 0)
        turnover = metrics.get('turnover_annual', 0)
        
        if turnover < 3:
            verdict_parts.append("   • Low turnover suitable for larger capital (₹50L - ₹2Cr)")
        elif turnover < 8:
            verdict_parts.append("   • Medium turnover suitable for ₹10L - ₹50L")
        else:
            verdict_parts.append("   • High turnover - keep capital under ₹25L")
        
        verdict_parts.append(f"   • Avg positions: {avg_positions:.1f} (diversification benefit)")
        
        # Structural weaknesses
        verdict_parts.append("\n3. STRUCTURAL WEAKNESSES:")
        
        weaknesses = []
        
        if max_dd < -0.30:
            weaknesses.append("   • High drawdown risk - needs stronger risk management")
        
        if win_rate < 0.45:
            weaknesses.append("   • Low win rate - requires patience and discipline")
        
        total_trades = metrics.get('total_trades', 0)
        if total_trades < 20:
            weaknesses.append("   • Low sample size - statistical significance questionable")
        
        if not weaknesses:
            weaknesses.append("   • No critical structural weaknesses identified")
        
        verdict_parts.extend(weaknesses)
        
        # Failure regimes
        verdict_parts.append("\n4. FAILURE REGIMES:")
        verdict_parts.append("   • Bear markets with momentum reversals")
        verdict_parts.append("   • High volatility with frequent stop-outs")
        verdict_parts.append("   • Low liquidity environments (slippage increase)")
        verdict_parts.append("   • Regime shifts (bull → bear transitions)")
        
        # Monte Carlo insights
        if monte_carlo_results:
            verdict_parts.append("\n5. ROBUSTNESS (Monte Carlo):")
            
            pct_better = 1 - monte_carlo_results.get('pct_better_than_original', 0.5)
            ci_low, ci_high = monte_carlo_results.get('cagr_confidence_95', (0, 0))
            
            verdict_parts.append(f"   • {pct_better:.0%} probability of beating original performance")
            verdict_parts.append(f"   • 95% CI: [{ci_low:.1%}, {ci_high:.1%}]")
            
            if pct_better > 0.30:
                verdict_parts.append("   ✓ Robust - performance not due to luck")
            else:
                verdict_parts.append("   ⚠ Questionable - may be due to lucky timing")
        
        # Final recommendation
        verdict_parts.append("\n6. FINAL RECOMMENDATION:")
        
        if sharpe > 1.5 and max_dd > -0.25 and (monte_carlo_results is None or pct_better > 0.30):
            verdict_parts.append("   ✓ APPROVED FOR LIVE TRADING (with position limits)")
        elif sharpe > 1.0:
            verdict_parts.append("   ⚠ APPROVED FOR PAPER TRADING FIRST")
        else:
            verdict_parts.append("   ✗ REQUIRES FURTHER OPTIMIZATION")
        
        verdict_parts.append("\n" + "="*60)
        
        return "\n".join(verdict_parts)
