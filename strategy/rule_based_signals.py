import statistics
from typing import List

class RuleBasedSignals:
    def __init__(self, logger=None):
        self.logger = logger

        # For scalping, we pick a small offset (in pips, ticks, or points).
        # Tune these to match your symbol’s average price movement & spread.
        self.SMALL_SL = 0.0010   # e.g. 0.001 points away
        self.SMALL_TP = 0.0015   # e.g. 0.0015 points away

        # Short windows for quick trend detection
        self.FAST_EMA_WINDOW = 3
        self.SLOW_EMA_WINDOW = 8

        # We want to quickly close if the market isn’t moving in our favor.
        self.SHORT_MAX_HOLDING_TIME = 3  # in minutes (or candles)

        # Store recent candle volumes and other candle data for later calculations.
        self.last_volumes: List[float] = []
        self.last_candle_times: List = []
        self.last_candle_opens: List[float] = []
        self.last_candle_highs: List[float] = []
        self.last_candle_lows: List[float] = []
        self.last_candle_closes: List[float] = []

        # To track trade outcomes: each element is a float profit (positive for wins, negative for losses)
        self.trade_results: List[float] = []

    def generate_candle_signal(
        self,
        candle_times: List,
        candle_opens: List[float],
        candle_highs: List[float],
        candle_lows: List[float],
        candle_closes: List[float],
        candle_volumes: List[float],
        recent_trades_win_loss_ratio: float = 1.0,
        account_balance: float = 1000.0
    ):
        # Save the latest candle data for later use.
        self.last_candle_times = candle_times
        self.last_candle_opens = candle_opens
        self.last_candle_highs = candle_highs
        self.last_candle_lows = candle_lows
        self.last_candle_closes = candle_closes
        self.last_volumes = candle_volumes

        # 1) Ensure we have enough candles to compute our short EMAs
        if len(candle_closes) < self.SLOW_EMA_WINDOW:
            if self.logger:
                self.logger.debug("Not enough candle data for short EMAs. Returning HOLD.")
            return self.default_hold_signal()

        # 2) Compute short EMAs
        fast_ema = self.compute_ema(candle_closes, self.FAST_EMA_WINDOW)
        slow_ema = self.compute_ema(candle_closes, self.SLOW_EMA_WINDOW)

        if fast_ema is None or slow_ema is None:
            if self.logger:
                self.logger.debug("Unable to compute short EMAs. Returning HOLD.")
            return self.default_hold_signal()

        current_close = candle_closes[-1]

        # 3) Determine direction
        if abs(fast_ema - slow_ema) < 1e-8:
            signal = "HOLD"
        elif fast_ema > slow_ema:
            signal = "BUY"
        else:
            signal = "SELL"

        # 4) If HOLD, just return
        if signal == "HOLD":
            return self.default_hold_signal()

        # 5) Define small stop-loss & take-profit around current_close
        if signal == "BUY":
            stop_loss = current_close - self.SMALL_SL
            take_profit = current_close + self.SMALL_TP
        else:  # SELL
            stop_loss = current_close + self.SMALL_SL
            take_profit = current_close - self.SMALL_TP

        # 6) Confidence is set to 1.0 for frequent trading
        confidence = 1.0

        # 7) Position size: small fraction (like 1%) of account for frequent trades
        position_size = account_balance * 0.01
        if position_size < 1.0:
            position_size = 1.0  # ensure a minimum stake

        # 8) Build final result
        result = {
            "signal": signal,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence": confidence,
            "position_size": position_size,
            "max_holding_time": self.SHORT_MAX_HOLDING_TIME,
            "trailing_stop": 0.0
        }

        if self.logger:
            self.logger.info(f"Scalping-based rule signal: {result}")
        return result

    def compute_ema(self, data: List[float], window: int):
        """
        Basic EMA calculation over entire series, returning the latest EMA value.
        """
        if len(data) < window:
            return None
        multiplier = 2 / (window + 1)
        ema_val = sum(data[:window]) / window  # starting with SMA for first window
        for p in data[window:]:
            ema_val = (p - ema_val) * multiplier + ema_val
        return ema_val

    def default_hold_signal(self):
        return {
            "signal": "HOLD",
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "confidence": 0.0,
            "position_size": 0.0,
            "max_holding_time": 1,
            "trailing_stop": 0.0
        }

    def calculate_recent_volume(self, window: int = 5) -> float:
        """
        Calculate the recent volume over the last `window` periods.
        If fewer than `window` data points are available, sums all available volumes.
        Returns the sum as a float.
        """
        if not self.last_volumes:
            if self.logger:
                self.logger.debug("No volume data available. Returning 0.")
            return 0.0

        if len(self.last_volumes) < window:
            recent_vol = sum(self.last_volumes)
        else:
            recent_vol = sum(self.last_volumes[-window:])

        if self.logger:
            self.logger.info(f"Calculated recent volume (last {window} periods): {recent_vol}")
        return recent_vol

    # New helper methods to access candle data
    def get_candle_times(self) -> List:
        """Return the list of stored candle timestamps."""
        return self.last_candle_times

    def get_candle_opens(self) -> List[float]:
        """Return the list of stored candle open prices."""
        return self.last_candle_opens

    def get_candle_highs(self) -> List[float]:
        """Return the list of stored candle high prices."""
        return self.last_candle_highs

    def get_candle_lows(self) -> List[float]:
        """Return the list of stored candle low prices."""
        return self.last_candle_lows

    def get_candle_closes(self) -> List[float]:
        """Return the list of stored candle close prices."""
        return self.last_candle_closes

    def record_trade_result(self, profit: float):
        """
        Record the profit (or loss) from a trade.
        Positive profit counts as a win; negative as a loss.
        """
        self.trade_results.append(profit)
        if self.logger:
            self.logger.info(f"Recorded trade result: {profit}")

    def calculate_recent_win_loss_ratio(self, window: int = 5) -> float:
        
        if not self.trade_results:
            if self.logger:
                self.logger.debug("No trade results recorded. Returning 1.0 as win/loss ratio.")
            return 1.0

        recent_trades = self.trade_results[-window:] if len(self.trade_results) >= window else self.trade_results
        wins = sum(1 for profit in recent_trades if profit > 0)
        losses = sum(1 for profit in recent_trades if profit < 0)

        if wins == 0 and losses == 0:
            return 1.0  # neutral
        if losses == 0:
            return float(wins)
        return wins / losses
    
    
    def calculate_volatility(self, prices: List[float], annualize: bool = False) -> float:
            if len(prices) < 2:
                if self.logger:
                    self.logger.debug("Not enough price data to calculate volatility. Returning 0.")
                return 0.0

            # Calculate percentage returns for consecutive prices.
            returns = []
            for i in range(1, len(prices)):
                prev = prices[i - 1]
                if prev != 0:
                    ret = (prices[i] - prev) / prev
                    returns.append(ret)
            if not returns:
                return 0.0

            vol = statistics.pstdev(returns)
            if annualize:
                # For example, if using daily data, multiply by sqrt(252)
                # If using minute data, adjust accordingly.
                # Here we assume daily; adjust factor as needed.
                vol *= (252 ** 0.5)

            if self.logger:
                self.logger.info(f"Calculated volatility: {vol}")
            return vol