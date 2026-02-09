"""
Visualization Module
====================

Professional charts and plots for strategy analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class StrategyVisualizer:
    """Create professional visualizations for strategy performance"""
    
    def __init__(self, figsize_base=(12, 6)):
        self.figsize_base = figsize_base
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'positive': '#06A77D',
            'negative': '#D81159',
            'neutral': '#8F8F8F'
        }
    
    def plot_equity_curve(
        self,
        equity_curve: pd.DataFrame,
        title: str = "Strategy Equity Curve",
        save_path: str = None
    ):
        """
        Plot equity curve with drawdown overlay.
        """
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        # Equity curve
        ax1.plot(equity_curve.index, equity_curve['equity'], 
                linewidth=2, color=self.colors['primary'], label='Portfolio Value')
        ax1.axhline(y=equity_curve['equity'].iloc[0], 
                   color=self.colors['neutral'], linestyle='--', 
                   linewidth=1, alpha=0.5, label='Initial Capital')
        
        ax1.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Portfolio Value (₹)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(style='plain', axis='y')
        
        # Format y-axis for currency
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'₹{x/100000:.1f}L')
        )
        
        # Drawdown
        equity_curve['cummax'] = equity_curve['equity'].cummax()
        equity_curve['drawdown'] = (equity_curve['equity'] - equity_curve['cummax']) / equity_curve['cummax']
        
        ax2.fill_between(equity_curve.index, 0, equity_curve['drawdown'] * 100,
                        color=self.colors['negative'], alpha=0.3)
        ax2.plot(equity_curve.index, equity_curve['drawdown'] * 100,
                color=self.colors['negative'], linewidth=1.5)
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([equity_curve['drawdown'].min() * 110, 5])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_comparison(
        self,
        original_equity: pd.DataFrame,
        rebuilt_equity: pd.DataFrame,
        save_path: str = None
    ):
        """
        Compare original vs rebuilt strategy performance.
        """
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Normalize to same starting point
        original_norm = original_equity['equity'] / original_equity['equity'].iloc[0] * 100
        rebuilt_norm = rebuilt_equity['equity'] / rebuilt_equity['equity'].iloc[0] * 100
        
        ax.plot(original_norm.index, original_norm.values,
               linewidth=2.5, color=self.colors['secondary'], 
               label='Original (No Costs)', alpha=0.8)
        
        ax.plot(rebuilt_norm.index, rebuilt_norm.values,
               linewidth=2.5, color=self.colors['primary'],
               label='Rebuilt (With Costs)', alpha=0.8)
        
        ax.axhline(y=100, color=self.colors['neutral'], 
                  linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_title('Strategy Comparison: Original vs Institutional Rebuild',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Normalized Return (Base = 100)', fontsize=12)
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_rolling_sharpe(
        self,
        equity_curve: pd.DataFrame,
        window: int = 252,
        save_path: str = None
    ):
        """
        Plot rolling Sharpe ratio.
        """
        
        fig, ax = plt.subplots(figsize=self.figsize_base)
        
        # Calculate rolling Sharpe
        equity_curve['returns'] = equity_curve['equity'].pct_change()
        rolling_mean = equity_curve['returns'].rolling(window).mean()
        rolling_std = equity_curve['returns'].rolling(window).std()
        
        rf_daily = 0.07 / 252
        rolling_sharpe = (rolling_mean - rf_daily) / rolling_std * np.sqrt(252)
        
        ax.plot(rolling_sharpe.index, rolling_sharpe.values,
               linewidth=2, color=self.colors['primary'])
        ax.axhline(y=0, color=self.colors['neutral'], 
                  linestyle='-', linewidth=1, alpha=0.3)
        ax.axhline(y=1, color=self.colors['positive'], 
                  linestyle='--', linewidth=1, alpha=0.5, label='Sharpe = 1.0')
        
        ax.set_title(f'Rolling {window}-Day Sharpe Ratio',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Sharpe Ratio', fontsize=11)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_return_distribution(
        self,
        trade_log: List[Dict],
        save_path: str = None
    ):
        """
        Plot distribution of trade returns.
        """
        
        if not trade_log:
            print("No trades to plot")
            return None
        
        trades_df = pd.DataFrame(trade_log)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(trades_df['return_net'] * 100, bins=30,
                color=self.colors['primary'], alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color=self.colors['negative'], 
                   linestyle='--', linewidth=2, label='Break-even')
        ax1.axvline(x=trades_df['return_net'].mean() * 100,
                   color=self.colors['positive'], linestyle='--', 
                   linewidth=2, label=f"Mean: {trades_df['return_net'].mean()*100:.2f}%")
        
        ax1.set_title('Distribution of Trade Returns',
                     fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Return (%)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Box plot
        ax2.boxplot([trades_df[trades_df['return_net'] > 0]['return_net'] * 100,
                    trades_df[trades_df['return_net'] < 0]['return_net'] * 100],
                   labels=['Winners', 'Losers'],
                   patch_artist=True,
                   boxprops=dict(facecolor=self.colors['primary'], alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        
        ax2.axhline(y=0, color=self.colors['neutral'], 
                   linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_title('Winner vs Loser Distribution',
                     fontsize=14, fontweight='bold', pad=15)
        ax2.set_ylabel('Return (%)', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_monthly_returns_heatmap(
        self,
        equity_curve: pd.DataFrame,
        save_path: str = None
    ):
        """
        Create monthly returns heatmap.
        """
        
        # Calculate monthly returns
        monthly_equity = equity_curve['equity'].resample('M').last()
        monthly_returns = monthly_equity.pct_change()
        
        # Create pivot table
        returns_pivot = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values * 100
        })
        
        returns_matrix = returns_pivot.pivot(
            index='Month', columns='Year', values='Return'
        )
        
        # Month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        returns_matrix.index = [month_names[i-1] for i in returns_matrix.index]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.heatmap(returns_matrix, annot=True, fmt='.1f', 
                   cmap='RdYlGn', center=0, 
                   cbar_kws={'label': 'Return (%)'},
                   linewidths=0.5, ax=ax)
        
        ax.set_title('Monthly Returns Heatmap',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Month', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_parameter_sensitivity(
        self,
        sensitivity_df: pd.DataFrame,
        save_path: str = None
    ):
        """
        Visualize parameter sensitivity analysis.
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Pivot for heatmaps
        metrics = ['cagr', 'sharpe', 'max_dd', 'win_rate']
        titles = ['CAGR (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            pivot = sensitivity_df.pivot(
                index='holding_days',
                columns='breakout_pct',
                values=metric
            )
            
            # Convert to percentages where appropriate
            if metric in ['cagr', 'max_dd', 'win_rate']:
                pivot = pivot * 100
            
            sns.heatmap(pivot, annot=True, fmt='.1f', 
                       cmap='YlGnBu' if metric != 'max_dd' else 'RdYlGn_r',
                       ax=ax, cbar_kws={'label': title})
            
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel('Breakout Threshold', fontsize=10)
            ax.set_ylabel('Holding Period (days)', fontsize=10)
            
            # Format x-axis labels as percentages
            xlabels = [f'{float(label.get_text())*100:.0f}%' 
                      for label in ax.get_xticklabels()]
            ax.set_xticklabels(xlabels)
        
        plt.suptitle('Parameter Sensitivity Analysis', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_monte_carlo_distribution(
        self,
        mc_results: Dict,
        save_path: str = None
    ):
        """
        Plot Monte Carlo simulation results.
        """
        
        if 'sharpe_distribution' not in mc_results:
            print("Monte Carlo results incomplete")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # CAGR distribution
        original_cagr = mc_results['original_cagr']
        mean_cagr = mc_results['mean_cagr']
        
        # Reconstruct distribution from percentiles (approximation)
        # In practice, you'd pass the full distribution
        cagr_values = []
        for p, v in mc_results['cagr_percentiles'].items():
            cagr_values.extend([v] * 10)  # Approximate
        
        ax1.hist(cagr_values, bins=30, color=self.colors['primary'], 
                alpha=0.7, edgecolor='black', density=True)
        ax1.axvline(x=original_cagr*100, color=self.colors['negative'],
                   linestyle='--', linewidth=2, 
                   label=f'Original: {original_cagr*100:.1f}%')
        ax1.axvline(x=mean_cagr*100, color=self.colors['positive'],
                   linestyle='--', linewidth=2,
                   label=f'Mean: {mean_cagr*100:.1f}%')
        
        ax1.set_title('Monte Carlo: CAGR Distribution',
                     fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('CAGR (%)', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Sharpe distribution
        if 'sharpe_distribution' in mc_results:
            sharpe_dist = mc_results['sharpe_distribution']
            
            ax2.hist(sharpe_dist, bins=30, color=self.colors['primary'],
                    alpha=0.7, edgecolor='black')
            ax2.axvline(x=np.mean(sharpe_dist), color=self.colors['positive'],
                       linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(sharpe_dist):.2f}')
            
            ax2.set_title('Monte Carlo: Sharpe Ratio Distribution',
                         fontsize=14, fontweight='bold', pad=15)
            ax2.set_xlabel('Sharpe Ratio', fontsize=11)
            ax2.set_ylabel('Frequency', fontsize=11)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_regime_performance(
        self,
        regime_df: pd.DataFrame,
        save_path: str = None
    ):
        """
        Visualize performance across market regimes.
        """
        
        if regime_df.empty:
            print("No regime data to plot")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Win rate by regime
        regime_df_sorted = regime_df.sort_values('win_rate', ascending=True)
        
        ax1.barh(regime_df_sorted['regime'], regime_df_sorted['win_rate'] * 100,
                color=self.colors['primary'], alpha=0.7)
        ax1.axvline(x=50, color=self.colors['neutral'], 
                   linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_xlabel('Win Rate (%)', fontsize=11)
        ax1.set_title('Win Rate by Market Regime',
                     fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Average return by regime
        regime_df_sorted2 = regime_df.sort_values('avg_return', ascending=True)
        colors = [self.colors['positive'] if x > 0 else self.colors['negative'] 
                 for x in regime_df_sorted2['avg_return']]
        
        ax2.barh(regime_df_sorted2['regime'], regime_df_sorted2['avg_return'] * 100,
                color=colors, alpha=0.7)
        ax2.axvline(x=0, color=self.colors['neutral'],
                   linestyle='-', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Average Return (%)', fontsize=11)
        ax2.set_title('Average Return by Market Regime',
                     fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
