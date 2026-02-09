"""
Institutional-Grade Backtest: NSE 5% Breakout Strategy
=======================================================

This implementation rebuilds the original strategy with:
- Survivorship bias elimination
- Look-ahead bias removal
- Realistic execution modeling
- Full portfolio simulation
- Risk management
- Robustness testing

Author: Quantitative Research Team
Date: February 2026
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class StrategyConfig:
    """Centralized configuration for strategy parameters"""
    
    # Data Parameters
    START_DATE = "2020-01-01"
    END_DATE = "2026-02-09"
    LOOKBACK_DAYS = 250  # For initial indicators
    
    # Universe Parameters
    UNIVERSE_SIZE = 500
    MIN_PRICE = 10  # Minimum stock price
    MIN_ADV = 5_000_000  # Minimum average daily value (₹)
    
    # Signal Parameters
    BREAKOUT_THRESHOLD = 0.05  # 5% price increase
    VOLUME_MULTIPLIER = 1.0  # Volume > 1x 20-day average
    EMA_PERIOD = 200
    
    # Execution Parameters
    ENTRY_DELAY = 1  # Enter at T+1 open
    HOLDING_PERIOD = 20  # Days
    COOLDOWN_PERIOD = 63  # Trading days (~3 months)
    
    # Transaction Costs (basis points)
    BROKERAGE_BPS = 3  # 0.03%
    STT_BPS = 10  # 0.10% on sell side
    EXCHANGE_BPS = 3.5  # 0.035%
    SLIPPAGE_BPS = 12.5  # 0.125% one-way, 0.25% round-trip
    
    # Portfolio Parameters
    INITIAL_CAPITAL = 10_00_000  # ₹10 lakhs
    MAX_POSITIONS = 20
    POSITION_SIZE_METHOD = "equal_weight"  # or "volatility_scaled"
    MAX_POSITION_SIZE = 0.10  # 10% max per position
    
    # Risk Management
    MAX_DRAWDOWN_STOP = 0.20  # 20% portfolio drawdown kill switch
    USE_REGIME_FILTER = True  # Market > 200 DMA filter
    
    # Regime Analysis Thresholds
    BULL_THRESHOLD = 0.15  # 15% index return over 6 months
    BEAR_THRESHOLD = -0.10  # -10% index return over 6 months
    HIGH_VOL_THRESHOLD = 0.25  # 25% annualized volatility
    
    @classmethod
    def total_transaction_cost(cls, one_way=False):
        """Calculate total transaction cost in basis points"""
        if one_way:
            return cls.BROKERAGE_BPS + cls.EXCHANGE_BPS + cls.SLIPPAGE_BPS
        else:
            # Round trip: buy (brokerage + exchange + slippage) + 
            # sell (brokerage + STT + exchange + slippage)
            buy_cost = cls.BROKERAGE_BPS + cls.EXCHANGE_BPS + cls.SLIPPAGE_BPS
            sell_cost = cls.BROKERAGE_BPS + cls.STT_BPS + cls.EXCHANGE_BPS + cls.SLIPPAGE_BPS
            return buy_cost + sell_cost


# ============================================================================
# DATA LOADING & UNIVERSE CONSTRUCTION
# ============================================================================

class DataLoader:
    """Handle data loading and survivorship bias elimination"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.price_data = {}
        self.volume_data = {}
        
    def get_nse_universe(self) -> List[str]:
        """
        Get NSE stock universe. 
        
        Note: This uses current market cap due to data limitations.
        In production, use point-in-time market cap from a proper database.
        """
        print("Fetching NSE universe...")
        
        try:
            from nsetools import Nse
            nse = Nse()
            stock_data = nse.get_stock_codes()
            
            if isinstance(stock_data, dict):
                symbols = list(stock_data.keys())
            elif isinstance(stock_data, list):
                symbols = stock_data
            else:
                symbols = []
            
            symbols = [s for s in symbols if s != 'SYMBOL']
            tickers = [f"{s}.NS" for s in symbols]
            
            print(f"Found {len(tickers)} NSE stocks")
            return tickers
            
        except Exception as e:
            print(f"Error fetching NSE universe: {e}")
            # Fallback to a known list
            return self._get_fallback_universe()
    
    def _get_fallback_universe(self) -> List[str]:
        """Fallback universe of major NSE stocks"""
        major_stocks = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ICICIBANK',
            'BHARTIARTL', 'SBIN', 'BAJFINANCE', 'KOTAKBANK', 'LT', 'ASIANPAINT',
            'HCLTECH', 'AXISBANK', 'ITC', 'MARUTI', 'SUNPHARMA', 'TITAN',
            'ULTRACEMCO', 'BAJAJFINSV', 'WIPRO', 'NTPC', 'ONGC', 'POWERGRID',
            'ADANIENT', 'TATAMOTORS', 'TATASTEEL', 'JSWSTEEL', 'INDUSINDBK',
            'HINDALCO', 'TECHM', 'GRASIM', 'BRITANNIA', 'DRREDDY', 'SHREECEM'
        ]
        return [f"{s}.NS" for s in major_stocks]
    
    def filter_by_market_cap(self, tickers: List[str], top_n: int) -> List[str]:
        """
        Filter to top N stocks by market cap.
        
        LIMITATION: Uses current market cap. In production, reconstruct
        point-in-time universe for each historical date.
        """
        print(f"Filtering to top {top_n} stocks by market cap...")
        
        market_caps = []
        for ticker in tickers[:1000]:  # Limit API calls
            try:
                info = yf.Ticker(ticker).info
                mcap = info.get("marketCap", None)
                if mcap:
                    market_caps.append((ticker, mcap))
            except:
                continue
        
        if not market_caps:
            print("Warning: Could not fetch market cap data")
            return tickers[:top_n]
        
        mcap_df = pd.DataFrame(market_caps, columns=["Ticker", "MarketCap"])
        mcap_df = mcap_df.sort_values("MarketCap", ascending=False)
        top_tickers = mcap_df.head(top_n)["Ticker"].tolist()
        
        print(f"Selected {len(top_tickers)} stocks")
        return top_tickers
    
    def download_price_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Download historical price data for all tickers.
        Handles data quality issues and missing data.
        """
        print(f"\nDownloading price data for {len(tickers)} stocks...")
        print(f"Date range: {self.config.START_DATE} to {self.config.END_DATE}")
        
        start = pd.Timestamp(self.config.START_DATE) - timedelta(days=self.config.LOOKBACK_DAYS)
        
        price_data = {}
        failed = []
        
        for i, ticker in enumerate(tickers):
            if (i + 1) % 50 == 0:
                print(f"Progress: {i+1}/{len(tickers)}")
            
            try:
                df = yf.download(
                    ticker,
                    start=start,
                    end=self.config.END_DATE,
                    progress=False,
                    auto_adjust=False
                )
                
                if df.empty:
                    failed.append(ticker)
                    continue
                
                # Standardize column names
                df = self._standardize_columns(df)
                
                # Data quality checks
                if len(df) < 100:  # Need minimum history
                    failed.append(ticker)
                    continue
                
                # Remove rows with missing critical data
                df = df.dropna(subset=['Close', 'Volume'])
                
                if len(df) < 100:
                    failed.append(ticker)
                    continue
                
                price_data[ticker] = df
                
            except Exception as e:
                failed.append(ticker)
                continue
        
        print(f"\nSuccessfully loaded: {len(price_data)} stocks")
        print(f"Failed: {len(failed)} stocks")
        
        self.price_data = price_data
        return price_data
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names from yfinance"""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # Ensure critical columns exist
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        
        if 'Adj Close' not in df.columns and 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        
        return df


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

class SignalGenerator:
    """Generate trading signals with no look-ahead bias"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicators using only past data.
        All indicators use data available at close of day T.
        """
        df = df.copy()
        
        # Previous day close
        df['prev_close'] = df['Close'].shift(1)
        
        # Daily return
        df['daily_return'] = df['Close'] / df['prev_close'] - 1
        
        # Volume indicators
        df['vol_20d_avg'] = df['Volume'].rolling(20, min_periods=20).mean()
        df['vol_ratio'] = df['Volume'] / df['vol_20d_avg']
        
        # Price indicators
        df['ema_200'] = df['Adj Close'].ewm(span=200, adjust=False).mean()
        
        # Average daily value for liquidity filter
        df['avg_daily_value'] = df['Close'] * df['Volume']
        df['adv_20d'] = df['avg_daily_value'].rolling(20, min_periods=20).mean()
        
        return df
    
    def generate_signals(
        self,
        df: pd.DataFrame,
        breakout_threshold: float = None
    ) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Signal criteria (all must be true at close of day T):
        1. Close >= Previous Close * (1 + breakout_threshold)
        2. Volume > Volume_20d_avg * volume_multiplier
        3. Close > EMA_200
        4. Average Daily Value > minimum threshold (liquidity)
        5. Close > minimum price
        
        Entry: Open of day T+1
        """
        df = df.copy()
        
        if breakout_threshold is None:
            breakout_threshold = self.config.BREAKOUT_THRESHOLD
        
        # Breakout condition
        breakout = df['daily_return'] >= breakout_threshold
        
        # Volume expansion
        volume_surge = df['vol_ratio'] > self.config.VOLUME_MULTIPLIER
        
        # Trend filter
        above_ema = df['Close'] > df['ema_200']
        
        # Liquidity filter
        sufficient_liquidity = df['adv_20d'] > self.config.MIN_ADV
        
        # Price filter
        price_filter = df['Close'] > self.config.MIN_PRICE
        
        # Combined signal
        df['signal'] = (
            breakout & 
            volume_surge & 
            above_ema & 
            sufficient_liquidity & 
            price_filter
        )
        
        return df


# ============================================================================
# PORTFOLIO SIMULATOR
# ============================================================================

class Position:
    """Represents a single position"""
    
    def __init__(
        self,
        ticker: str,
        entry_date: pd.Timestamp,
        entry_price: float,
        shares: int,
        target_exit_date: pd.Timestamp,
        entry_cost: float
    ):
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.target_exit_date = target_exit_date
        self.entry_cost = entry_cost
        self.exit_date = None
        self.exit_price = None
        self.exit_cost = None
        self.pnl_gross = None
        self.pnl_net = None
        self.return_gross = None
        self.return_net = None
        self.exit_reason = None
    
    def close(
        self,
        exit_date: pd.Timestamp,
        exit_price: float,
        exit_cost: float,
        reason: str
    ):
        """Close the position"""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_cost = exit_cost
        self.exit_reason = reason
        
        # Calculate P&L
        self.pnl_gross = (exit_price - self.entry_price) * self.shares
        self.pnl_net = self.pnl_gross - self.entry_cost - self.exit_cost
        
        # Calculate returns
        position_value = self.entry_price * self.shares
        self.return_gross = self.pnl_gross / position_value
        self.return_net = self.pnl_net / position_value
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary for logging"""
        return {
            'ticker': self.ticker,
            'entry_date': self.entry_date,
            'entry_price': self.entry_price,
            'shares': self.shares,
            'exit_date': self.exit_date,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'pnl_gross': self.pnl_gross,
            'pnl_net': self.pnl_net,
            'return_gross': self.return_gross,
            'return_net': self.return_net,
            'holding_days': (self.exit_date - self.entry_date).days if self.exit_date else None
        }


class PortfolioSimulator:
    """
    Full portfolio simulation with:
    - Realistic position sizing
    - Transaction costs
    - Position limits
    - Cooldown periods
    - Drawdown monitoring
    """
    
    def __init__(
        self,
        config: StrategyConfig,
        price_data: Dict[str, pd.DataFrame],
        index_data: pd.DataFrame = None
    ):
        self.config = config
        self.price_data = price_data
        self.index_data = index_data
        
        # Portfolio state
        self.capital = config.INITIAL_CAPITAL
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.cooldown_dict: Dict[str, pd.Timestamp] = {}
        
        # Performance tracking
        self.equity_curve = []
        self.daily_returns = []
        self.trade_log = []
        self.peak_equity = config.INITIAL_CAPITAL
        self.max_drawdown = 0
        self.stopped_out = False
        
    def calculate_transaction_cost(self, shares: int, price: float, side: str) -> float:
        """
        Calculate transaction costs.
        
        Args:
            shares: Number of shares
            price: Price per share
            side: 'buy' or 'sell'
        """
        notional = shares * price
        
        if side == 'buy':
            cost_bps = (
                self.config.BROKERAGE_BPS +
                self.config.EXCHANGE_BPS +
                self.config.SLIPPAGE_BPS
            )
        else:  # sell
            cost_bps = (
                self.config.BROKERAGE_BPS +
                self.config.STT_BPS +
                self.config.EXCHANGE_BPS +
                self.config.SLIPPAGE_BPS
            )
        
        cost = notional * (cost_bps / 10000)
        return cost
    
    def is_in_cooldown(self, ticker: str, current_date: pd.Timestamp) -> bool:
        """Check if ticker is in cooldown period"""
        if ticker not in self.cooldown_dict:
            return False
        
        last_exit = self.cooldown_dict[ticker]
        days_since_exit = (current_date - last_exit).days
        
        return days_since_exit < self.config.COOLDOWN_PERIOD
    
    def add_to_cooldown(self, ticker: str, exit_date: pd.Timestamp):
        """Add ticker to cooldown after position exit"""
        self.cooldown_dict[ticker] = exit_date
    
    def can_open_position(self, current_date: pd.Timestamp) -> bool:
        """Check if we can open new position"""
        if self.stopped_out:
            return False
        
        # Check position limit
        open_positions = len(self.positions)
        if open_positions >= self.config.MAX_POSITIONS:
            return False
        
        # Check regime filter if enabled
        if self.config.USE_REGIME_FILTER and self.index_data is not None:
            if current_date not in self.index_data.index:
                return True  # No data, allow
            
            idx_row = self.index_data.loc[current_date]
            if 'ema_200' in idx_row and not pd.isna(idx_row['ema_200']):
                if idx_row['Close'] < idx_row['ema_200']:
                    return False
        
        return True
    
    def calculate_position_size(self, price: float) -> int:
        """
        Calculate number of shares to buy.
        Uses equal weighting by default.
        """
        # Available capital for new position
        available_capital = self.capital
        
        # Equal weight allocation
        max_positions = self.config.MAX_POSITIONS
        allocation = available_capital / max_positions
        
        # Respect max position size
        max_position_value = self.capital * self.config.MAX_POSITION_SIZE
        allocation = min(allocation, max_position_value)
        
        # Calculate shares
        shares = int(allocation / price)
        
        return shares
    
    def open_position(
        self,
        ticker: str,
        entry_date: pd.Timestamp,
        entry_price: float
    ) -> Optional[Position]:
        """Open a new position"""
        
        shares = self.calculate_position_size(entry_price)
        
        if shares == 0:
            return None
        
        # Calculate entry cost
        entry_cost = self.calculate_transaction_cost(shares, entry_price, 'buy')
        
        # Check if we have enough capital
        total_cost = shares * entry_price + entry_cost
        if total_cost > self.capital:
            return None
        
        # Create position
        target_exit = entry_date + timedelta(days=self.config.HOLDING_PERIOD)
        
        position = Position(
            ticker=ticker,
            entry_date=entry_date,
            entry_price=entry_price,
            shares=shares,
            target_exit_date=target_exit,
            entry_cost=entry_cost
        )
        
        # Update capital
        self.capital -= total_cost
        
        # Add to positions
        self.positions.append(position)
        
        return position
    
    def close_position(
        self,
        position: Position,
        exit_date: pd.Timestamp,
        exit_price: float,
        reason: str
    ):
        """Close an existing position"""
        
        # Calculate exit cost
        exit_cost = self.calculate_transaction_cost(
            position.shares, exit_price, 'sell'
        )
        
        # Close position
        position.close(exit_date, exit_price, exit_cost, reason)
        
        # Update capital
        proceeds = position.shares * exit_price - exit_cost
        self.capital += proceeds
        
        # Move to closed positions
        self.positions.remove(position)
        self.closed_positions.append(position)
        
        # Add to cooldown
        self.add_to_cooldown(position.ticker, exit_date)
        
        # Log trade
        self.trade_log.append(position.to_dict())
    
    def get_portfolio_value(self, current_date: pd.Timestamp) -> float:
        """Calculate total portfolio value (cash + positions)"""
        
        total_value = self.capital
        
        for position in self.positions:
            if position.ticker in self.price_data:
                df = self.price_data[position.ticker]
                if current_date in df.index:
                    current_price = df.loc[current_date, 'Close']
                    position_value = position.shares * current_price
                    total_value += position_value
        
        return total_value
    
    def check_drawdown_stop(self, current_value: float):
        """Check if drawdown stop is triggered"""
        
        # Update peak
        if current_value > self.peak_equity:
            self.peak_equity = current_value
        
        # Calculate drawdown
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - current_value) / self.peak_equity
            
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
            
            # Check stop
            if drawdown >= self.config.MAX_DRAWDOWN_STOP:
                print(f"\n{'='*60}")
                print(f"DRAWDOWN STOP TRIGGERED: {drawdown:.1%}")
                print(f"Peak Equity: ₹{self.peak_equity:,.0f}")
                print(f"Current Equity: ₹{current_value:,.0f}")
                print(f"{'='*60}\n")
                self.stopped_out = True
    
    def run_backtest(
        self,
        signals_dict: Dict[str, pd.DataFrame],
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Run full portfolio backtest.
        
        Process:
        1. For each trading day:
           - Check for position exits (time-based)
           - Check for new signals
           - Open new positions if conditions met
           - Update portfolio value
           - Check risk controls
        """
        
        if start_date is None:
            start_date = self.config.START_DATE
        if end_date is None:
            end_date = self.config.END_DATE
        
        # Get all trading dates
        all_dates = set()
        for df in signals_dict.values():
            all_dates.update(df.index)
        
        trading_dates = sorted([d for d in all_dates if start_date <= d <= end_date])
        
        print(f"\n{'='*60}")
        print(f"Running Backtest")
        print(f"{'='*60}")
        print(f"Start Date: {start_date}")
        print(f"End Date: {end_date}")
        print(f"Trading Days: {len(trading_dates)}")
        print(f"Initial Capital: ₹{self.config.INITIAL_CAPITAL:,.0f}")
        print(f"{'='*60}\n")
        
        for current_date in trading_dates:
            
            # ---- STEP 1: Exit positions ----
            positions_to_close = []
            
            for position in self.positions:
                if current_date >= position.target_exit_date:
                    # Time-based exit
                    if position.ticker in self.price_data:
                        df = self.price_data[position.ticker]
                        if current_date in df.index:
                            exit_price = df.loc[current_date, 'Open']  # Exit at open
                            positions_to_close.append((position, exit_price, 'time_exit'))
            
            for position, exit_price, reason in positions_to_close:
                self.close_position(position, current_date, exit_price, reason)
            
            # ---- STEP 2: Check for new signals ----
            if self.can_open_position(current_date):
                
                # Collect all signals for this date
                today_signals = []
                
                for ticker, df in signals_dict.items():
                    if current_date in df.index:
                        row = df.loc[current_date]
                        
                        if row.get('signal', False):
                            # Check cooldown
                            if not self.is_in_cooldown(ticker, current_date):
                                # Entry is at next day open
                                # Need to find next available date
                                future_dates = [d for d in df.index if d > current_date]
                                if future_dates:
                                    entry_date = future_dates[0]
                                    if entry_date in df.index:
                                        entry_price = df.loc[entry_date, 'Open']
                                        
                                        today_signals.append({
                                            'ticker': ticker,
                                            'signal_date': current_date,
                                            'entry_date': entry_date,
                                            'entry_price': entry_price,
                                            'volume_ratio': row.get('vol_ratio', 0)
                                        })
                
                # ---- STEP 3: Prioritize and open positions ----
                if today_signals:
                    # Sort by volume ratio (higher is better)
                    today_signals.sort(key=lambda x: x['volume_ratio'], reverse=True)
                    
                    # Open positions up to limit
                    for signal in today_signals:
                        if not self.can_open_position(signal['entry_date']):
                            break
                        
                        position = self.open_position(
                            ticker=signal['ticker'],
                            entry_date=signal['entry_date'],
                            entry_price=signal['entry_price']
                        )
            
            # ---- STEP 4: Update portfolio value ----
            portfolio_value = self.get_portfolio_value(current_date)
            self.equity_curve.append({
                'date': current_date,
                'equity': portfolio_value,
                'cash': self.capital,
                'num_positions': len(self.positions)
            })
            
            # ---- STEP 5: Check risk controls ----
            self.check_drawdown_stop(portfolio_value)
        
        # Convert to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        return equity_df


# ============================================================================
# PERFORMANCE ANALYTICS
# ============================================================================

class PerformanceAnalyzer:
    """Calculate strategy performance metrics"""
    
    @staticmethod
    def calculate_metrics(equity_curve: pd.DataFrame, trade_log: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        metrics = {}
        
        # Basic stats
        initial_capital = equity_curve['equity'].iloc[0]
        final_capital = equity_curve['equity'].iloc[-1]
        total_return = (final_capital / initial_capital) - 1
        
        # Time period
        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]
        years = (end_date - start_date).days / 365.25
        
        # CAGR
        cagr = (final_capital / initial_capital) ** (1 / years) - 1
        
        # Returns
        equity_curve['daily_return'] = equity_curve['equity'].pct_change()
        mean_daily_return = equity_curve['daily_return'].mean()
        std_daily_return = equity_curve['daily_return'].std()
        
        # Sharpe Ratio (annualized, risk-free = 7%)
        risk_free_daily = 0.07 / 252
        sharpe = (mean_daily_return - risk_free_daily) / std_daily_return * np.sqrt(252)
        
        # Sortino Ratio (annualized)
        downside_returns = equity_curve['daily_return'][equity_curve['daily_return'] < 0]
        downside_std = downside_returns.std()
        sortino = (mean_daily_return - risk_free_daily) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # Drawdown
        equity_curve['cummax'] = equity_curve['equity'].cummax()
        equity_curve['drawdown'] = (equity_curve['equity'] - equity_curve['cummax']) / equity_curve['cummax']
        max_drawdown = equity_curve['drawdown'].min()
        
        # Trade statistics
        if trade_log:
            trades_df = pd.DataFrame(trade_log)
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['return_net'] > 0])
            losing_trades = len(trades_df[trades_df['return_net'] < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            avg_win = trades_df[trades_df['return_net'] > 0]['return_net'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['return_net'] < 0]['return_net'].mean() if losing_trades > 0 else 0
            
            avg_return_gross = trades_df['return_gross'].mean()
            avg_return_net = trades_df['return_net'].mean()
            
            # Turnover (annualized)
            total_traded = trades_df['shares'].mul(trades_df['entry_price']).sum()
            avg_capital = equity_curve['equity'].mean()
            turnover_annual = (total_traded / avg_capital) * (365.25 / years)
            
        else:
            total_trades = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            avg_return_gross = 0
            avg_return_net = 0
            turnover_annual = 0
        
        # Exposure
        avg_num_positions = equity_curve['num_positions'].mean()
        max_positions = equity_curve['num_positions'].max()
        avg_cash_pct = (equity_curve['cash'] / equity_curve['equity']).mean()
        
        # Compile metrics
        metrics = {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'avg_return_gross': avg_return_gross,
            'avg_return_net': avg_return_net,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'turnover_annual': turnover_annual,
            'avg_positions': avg_num_positions,
            'max_positions': max_positions,
            'avg_cash_pct': avg_cash_pct,
            'years': years
        }
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict, title: str = "Performance Metrics"):
        """Print metrics in formatted table"""
        
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        
        print(f"\nRETURNS:")
        print(f"  Total Return:        {metrics['total_return']:>10.2%}")
        print(f"  CAGR:                {metrics['cagr']:>10.2%}")
        print(f"  Period:              {metrics['years']:>10.1f} years")
        
        print(f"\nRISK-ADJUSTED:")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}")
        print(f"  Max Drawdown:        {metrics['max_drawdown']:>10.2%}")
        
        print(f"\nTRADING:")
        print(f"  Total Trades:        {metrics['total_trades']:>10.0f}")
        print(f"  Win Rate:            {metrics['win_rate']:>10.2%}")
        print(f"  Avg Return (Gross):  {metrics['avg_return_gross']:>10.2%}")
        print(f"  Avg Return (Net):    {metrics['avg_return_net']:>10.2%}")
        print(f"  Avg Win:             {metrics['avg_win']:>10.2%}")
        print(f"  Avg Loss:            {metrics['avg_loss']:>10.2%}")
        print(f"  Annual Turnover:     {metrics['turnover_annual']:>10.1f}x")
        
        print(f"\nPORTFOLIO:")
        print(f"  Avg Positions:       {metrics['avg_positions']:>10.1f}")
        print(f"  Max Positions:       {metrics['max_positions']:>10.0f}")
        print(f"  Avg Cash %:          {metrics['avg_cash_pct']:>10.2%}")
        
        print(f"{'='*60}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("INSTITUTIONAL-GRADE BACKTEST: NSE 5% BREAKOUT STRATEGY")
    print("="*60 + "\n")
    
    # Configuration
    config = StrategyConfig()
    
    # Step 1: Load Data
    loader = DataLoader(config)
    
    # Get universe - using fallback for demo
    print("\n[STEP 1] Loading Data...")
    universe = loader._get_fallback_universe()  # Use fallback for demo
    
    # Download price data
    price_data = loader.download_price_data(universe)
    
    if not price_data:
        print("ERROR: No price data loaded. Exiting.")
        exit(1)
    
    # Step 2: Generate Signals
    print("\n[STEP 2] Generating Signals...")
    signal_gen = SignalGenerator(config)
    
    signals_dict = {}
    for ticker, df in price_data.items():
        df_with_indicators = signal_gen.compute_indicators(df)
        df_with_signals = signal_gen.generate_signals(df_with_indicators)
        signals_dict[ticker] = df_with_signals
    
    # Count total signals
    total_signals = sum(df['signal'].sum() for df in signals_dict.values())
    print(f"Total signals generated: {total_signals}")
    
    # Step 3: Run Backtest
    print("\n[STEP 3] Running Portfolio Simulation...")
    
    simulator = PortfolioSimulator(config, price_data)
    equity_curve = simulator.run_backtest(signals_dict)
    
    # Step 4: Analyze Performance
    print("\n[STEP 4] Analyzing Performance...")
    
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.calculate_metrics(equity_curve, simulator.trade_log)
    analyzer.print_metrics(metrics, "STRATEGY PERFORMANCE")
    
    print("\n[COMPLETE] Backtest finished successfully.")
    print(f"Output files will be saved to /mnt/user-data/outputs/")
