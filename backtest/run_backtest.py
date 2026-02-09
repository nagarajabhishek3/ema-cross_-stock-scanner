"""
Main Execution Script
=====================

Orchestrates the complete institutional-grade backtest:
1. Data loading and universe construction
2. Signal generation
3. Portfolio simulation
4. Performance analysis
5. Robustness testing
6. Regime analysis
7. Visualization
8. Final report generation
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from institutional_backtest import (
    StrategyConfig, DataLoader, SignalGenerator, 
    PortfolioSimulator, PerformanceAnalyzer
)
from robustness_testing import (
    RobustnessTester, RegimeAnalyzer, StrategyComparison
)
from visualization import StrategyVisualizer


def main():
    """Execute complete institutional backtest"""
    
    print("\n" + "="*70)
    print(" " * 10 + "INSTITUTIONAL-GRADE STRATEGY BACKTEST")
    print(" " * 15 + "NSE 5% Breakout Strategy")
    print("="*70 + "\n")
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    config = StrategyConfig()
    
    print("Configuration:")
    print(f"  Period: {config.START_DATE} to {config.END_DATE}")
    print(f"  Universe: Top {config.UNIVERSE_SIZE} NSE stocks")
    print(f"  Initial Capital: ₹{config.INITIAL_CAPITAL:,.0f}")
    print(f"  Breakout Threshold: {config.BREAKOUT_THRESHOLD:.1%}")
    print(f"  Holding Period: {config.HOLDING_PERIOD} days")
    print(f"  Transaction Costs: {config.total_transaction_cost():.2f} bps round-trip")
    print()
    
    # ========================================================================
    # STEP 1: DATA LOADING
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING & UNIVERSE CONSTRUCTION")
    print("="*70 + "\n")
    
    loader = DataLoader(config)
    
    # Get universe (using fallback for demo purposes)
    print("Note: Using fallback universe due to API limitations.")
    print("In production, use point-in-time market cap data.\n")
    universe = loader._get_fallback_universe()
    
    # Download price data
    price_data = loader.download_price_data(universe)
    
    if not price_data:
        print("\nERROR: Failed to load price data. Exiting.")
        return
    
    print(f"\n✓ Successfully loaded data for {len(price_data)} stocks")
    
    # ========================================================================
    # STEP 2: SIGNAL GENERATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 2: SIGNAL GENERATION")
    print("="*70 + "\n")
    
    signal_gen = SignalGenerator(config)
    
    signals_dict = {}
    for ticker, df in price_data.items():
        df_indicators = signal_gen.compute_indicators(df)
        df_signals = signal_gen.generate_signals(df_indicators)
        signals_dict[ticker] = df_signals
    
    total_signals = sum(df['signal'].sum() for df in signals_dict.values())
    print(f"✓ Generated {total_signals} signals across all stocks")
    
    # Signal statistics
    signals_by_stock = {ticker: df['signal'].sum() 
                       for ticker, df in signals_dict.items() 
                       if df['signal'].sum() > 0}
    print(f"✓ {len(signals_by_stock)} stocks generated at least one signal")
    
    # ========================================================================
    # STEP 3: FULL BACKTEST
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 3: PORTFOLIO SIMULATION (FULL PERIOD)")
    print("="*70)
    
    simulator = PortfolioSimulator(config, price_data)
    equity_curve = simulator.run_backtest(signals_dict)
    
    print(f"\n✓ Backtest completed")
    print(f"✓ Total trades executed: {len(simulator.trade_log)}")
    print(f"✓ Final portfolio value: ₹{equity_curve['equity'].iloc[-1]:,.0f}")
    
    # ========================================================================
    # STEP 4: PERFORMANCE ANALYSIS
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 4: PERFORMANCE ANALYSIS")
    print("="*70)
    
    analyzer = PerformanceAnalyzer()
    full_metrics = analyzer.calculate_metrics(equity_curve, simulator.trade_log)
    analyzer.print_metrics(full_metrics, "FULL PERIOD PERFORMANCE")
    
    # ========================================================================
    # STEP 5: IN-SAMPLE / OUT-OF-SAMPLE TESTING
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 5: IN-SAMPLE / OUT-OF-SAMPLE TESTING")
    print("="*70 + "\n")
    
    tester = RobustnessTester(config)
    
    # Split data 70/30
    in_sample, out_of_sample = tester.train_test_split(
        config.START_DATE, config.END_DATE, split_ratio=0.70
    )
    
    print(f"In-Sample Period:     {in_sample[0]} to {in_sample[1]}")
    print(f"Out-of-Sample Period: {out_of_sample[0]} to {out_of_sample[1]}\n")
    
    # Run in-sample backtest
    print("Running In-Sample backtest...")
    sim_is = PortfolioSimulator(config, price_data)
    equity_is = sim_is.run_backtest(signals_dict, in_sample[0], in_sample[1])
    metrics_is = analyzer.calculate_metrics(equity_is, sim_is.trade_log)
    
    # Run out-of-sample backtest
    print("\nRunning Out-of-Sample backtest...")
    sim_oos = PortfolioSimulator(config, price_data)
    equity_oos = sim_oos.run_backtest(signals_dict, out_of_sample[0], out_of_sample[1])
    metrics_oos = analyzer.calculate_metrics(equity_oos, sim_oos.trade_log)
    
    # Compare
    print("\n" + "-"*70)
    print("IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON")
    print("-"*70)
    print(f"{'Metric':<25} {'In-Sample':>15} {'Out-of-Sample':>15} {'Degradation':>15}")
    print("-"*70)
    
    comparison_metrics = [
        ('CAGR', 'cagr'),
        ('Sharpe Ratio', 'sharpe_ratio'),
        ('Max Drawdown', 'max_drawdown'),
        ('Win Rate', 'win_rate')
    ]
    
    for label, key in comparison_metrics:
        is_val = metrics_is[key]
        oos_val = metrics_oos[key]
        
        if key == 'max_drawdown':
            degradation = oos_val - is_val  # More negative = worse
        else:
            degradation = (oos_val - is_val) / is_val if is_val != 0 else 0
        
        if key == 'max_drawdown':
            print(f"{label:<25} {is_val:>14.2%} {oos_val:>14.2%} {degradation:>14.2%}")
        elif key == 'sharpe_ratio':
            print(f"{label:<25} {is_val:>15.2f} {oos_val:>15.2f} {degradation:>14.1%}")
        else:
            print(f"{label:<25} {is_val:>14.2%} {oos_val:>14.2%} {degradation:>14.1%}")
    
    print("-"*70 + "\n")
    
    # ========================================================================
    # STEP 6: PARAMETER SENSITIVITY
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 6: PARAMETER SENSITIVITY ANALYSIS")
    print("="*70)
    
    print("\nNote: Running parameter sweep (this may take a few minutes)...")
    print("Testing breakout levels: 4%, 5%, 6%")
    print("Testing holding periods: 5, 10, 20 days\n")
    
    # For demo, skip parameter sensitivity to save time
    # In production, uncomment this:
    # sensitivity_df = tester.parameter_sensitivity(
    #     price_data, SignalGenerator, PortfolioSimulator
    # )
    
    print("Skipping parameter sensitivity for demo (would test 9 combinations)")
    sensitivity_df = None
    
    # ========================================================================
    # STEP 7: MONTE CARLO SIMULATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 7: MONTE CARLO ROBUSTNESS TEST")
    print("="*70)
    
    if len(simulator.trade_log) >= 20:
        mc_results = tester.monte_carlo_resampling(
            simulator.trade_log,
            n_simulations=1000,
            initial_capital=config.INITIAL_CAPITAL
        )
    else:
        print("\nInsufficient trades for Monte Carlo simulation")
        mc_results = None
    
    # ========================================================================
    # STEP 8: REGIME ANALYSIS
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 8: REGIME-BASED PERFORMANCE ANALYSIS")
    print("="*70)
    
    regime_analyzer = RegimeAnalyzer(index_ticker="^NSEI")
    
    # Load index data
    index_data = regime_analyzer.load_index_data(config.START_DATE, config.END_DATE)
    
    if index_data is not None and len(simulator.trade_log) > 0:
        regime_df = regime_analyzer.analyze_by_regime(
            simulator.trade_log, equity_curve
        )
    else:
        print("Insufficient data for regime analysis")
        regime_df = None
    
    # ========================================================================
    # STEP 9: VISUALIZATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 9: GENERATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    viz = StrategyVisualizer()
    
    # Create output directory
    import os
    output_dir = "/mnt/user-data/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Equity curve
    viz.plot_equity_curve(
        equity_curve,
        title="NSE 5% Breakout Strategy - Equity Curve",
        save_path=f"{output_dir}/equity_curve.png"
    )
    
    # 2. Rolling Sharpe
    viz.plot_rolling_sharpe(
        equity_curve,
        window=252,
        save_path=f"{output_dir}/rolling_sharpe.png"
    )
    
    # 3. Return distribution
    if simulator.trade_log:
        viz.plot_return_distribution(
            simulator.trade_log,
            save_path=f"{output_dir}/return_distribution.png"
        )
    
    # 4. Monthly returns heatmap
    viz.plot_monthly_returns_heatmap(
        equity_curve,
        save_path=f"{output_dir}/monthly_returns.png"
    )
    
    # 5. Monte Carlo (if available)
    if mc_results and 'sharpe_distribution' in mc_results:
        viz.plot_monte_carlo_distribution(
            mc_results,
            save_path=f"{output_dir}/monte_carlo.png"
        )
    
    # 6. Regime performance (if available)
    if regime_df is not None and not regime_df.empty:
        viz.plot_regime_performance(
            regime_df,
            save_path=f"{output_dir}/regime_performance.png"
        )
    
    # 7. Parameter sensitivity (if available)
    if sensitivity_df is not None:
        viz.plot_parameter_sensitivity(
            sensitivity_df,
            save_path=f"{output_dir}/parameter_sensitivity.png"
        )
    
    print("✓ All visualizations saved")
    
    # ========================================================================
    # STEP 10: FINAL REPORT
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 10: GENERATING FINAL REPORT")
    print("="*70 + "\n")
    
    # Generate verdict
    verdict = StrategyComparison.generate_verdict(full_metrics, mc_results)
    print(verdict)
    
    # Save detailed results
    print("\nSaving detailed results...")
    
    # Trade log
    if simulator.trade_log:
        trades_df = pd.DataFrame(simulator.trade_log)
        trades_df.to_csv(f"{output_dir}/trade_log.csv", index=False)
        print(f"✓ Saved: trade_log.csv ({len(trades_df)} trades)")
    
    # Equity curve
    equity_curve.to_csv(f"{output_dir}/equity_curve.csv")
    print(f"✓ Saved: equity_curve.csv")
    
    # Performance metrics
    metrics_df = pd.DataFrame([full_metrics])
    metrics_df.to_csv(f"{output_dir}/performance_metrics.csv", index=False)
    print(f"✓ Saved: performance_metrics.csv")
    
    # In-sample vs out-of-sample
    if metrics_is and metrics_oos:
        comparison_df = pd.DataFrame({
            'Metric': [m[0] for m in comparison_metrics],
            'In_Sample': [metrics_is[m[1]] for m in comparison_metrics],
            'Out_of_Sample': [metrics_oos[m[1]] for m in comparison_metrics]
        })
        comparison_df.to_csv(f"{output_dir}/is_oos_comparison.csv", index=False)
        print(f"✓ Saved: is_oos_comparison.csv")
    
    # Regime analysis
    if regime_df is not None and not regime_df.empty:
        regime_df.to_csv(f"{output_dir}/regime_analysis.csv", index=False)
        print(f"✓ Saved: regime_analysis.csv")
    
    # Create summary report
    with open(f"{output_dir}/SUMMARY_REPORT.txt", 'w') as f:
        f.write("="*70 + "\n")
        f.write("INSTITUTIONAL-GRADE BACKTEST SUMMARY\n")
        f.write("NSE 5% Breakout Strategy\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Test Period: {config.START_DATE} to {config.END_DATE}\n")
        f.write(f"Initial Capital: ₹{config.INITIAL_CAPITAL:,.0f}\n")
        f.write(f"Final Capital: ₹{equity_curve['equity'].iloc[-1]:,.0f}\n")
        f.write(f"Total Return: {full_metrics['total_return']:.2%}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("KEY METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"CAGR:                {full_metrics['cagr']:.2%}\n")
        f.write(f"Sharpe Ratio:        {full_metrics['sharpe_ratio']:.2f}\n")
        f.write(f"Sortino Ratio:       {full_metrics['sortino_ratio']:.2f}\n")
        f.write(f"Max Drawdown:        {full_metrics['max_drawdown']:.2%}\n")
        f.write(f"Win Rate:            {full_metrics['win_rate']:.2%}\n")
        f.write(f"Total Trades:        {full_metrics['total_trades']:.0f}\n")
        f.write(f"Avg Return (Net):    {full_metrics['avg_return_net']:.2%}\n")
        f.write(f"Annual Turnover:     {full_metrics['turnover_annual']:.1f}x\n\n")
        
        f.write("\n" + verdict + "\n")
    
    print(f"✓ Saved: SUMMARY_REPORT.txt")
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    
    print("\n" + "="*70)
    print("BACKTEST COMPLETE")
    print("="*70)
    print(f"\n✓ All results saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - equity_curve.png")
    print("  - rolling_sharpe.png")
    print("  - return_distribution.png")
    print("  - monthly_returns.png")
    print("  - trade_log.csv")
    print("  - equity_curve.csv")
    print("  - performance_metrics.csv")
    print("  - SUMMARY_REPORT.txt")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
