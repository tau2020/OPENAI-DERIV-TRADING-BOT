# strategy/rule_based_signals.py

import statistics
from typing import List
import math

class RuleBasedSignals:
    """
    A simple rule-based signal generator that calculates indicators and
    produces signals (BUY, SELL, HOLD) along with SL/TP.
    """
    def __init__(self, logger=None):
        self.logger = logger
        # You can define configurable thresholds
        self.MIN_ATR = 0.0001  # Avoid divide-by-zero
        self.RISK_MULTIPLIER_SL = 1.5
        self.RISK_MULTIPLIER_TP = 2.0
        self.CONFIDENCE_THRESHOLD_UPTREND = 0.8
        self.CONFIDENCE_THRESHOLD_DOWNTREND = 0.8

    def generate_signal(
        self,
        prices: List[float],
        times: List,
        volume_data: List[int] = None,
        volatility: float = 0.0,
        sentiment_score: float = 0.0,
        recent_trades_win_loss_ratio: float = 1.0,
        account_balance: float = 1000.0,
    ):
        """
        Replaces the AI approach with a rule-based approach.
        Returns a dictionary with keys:
            "signal", "stop_loss", "take_profit", "confidence",
            "position_size", "max_holding_time", "trailing_stop"
        """

        # 1. Compute indicators
        ema_20 = self.compute_ema(prices, 20)
        ema_50 = self.compute_ema(prices, 50)
        ema_200 = self.compute_ema(prices, 200)
        atr = self.compute_atr(prices, window=14)

        # 2. Determine trend logic
        if ema_20 and ema_50 and ema_200:
            # Uptrend if 20 > 50 > 200
            uptrend = (ema_20 > ema_50 > ema_200)
            # Downtrend if 20 < 50 < 200
            downtrend = (ema_20 < ema_50 < ema_200)
        else:
            uptrend = False
            downtrend = False

        if len(prices) == 0:
            # Not enough data, default to HOLD
            return self.default_hold_signal()

        current_price = prices[-1]

        # 3. Simple signal logic
        signal = "HOLD"
        if uptrend:
            # For demonstration, let's say if the current price is near the 20 EMA => BUY
            if current_price < (ema_20 * 1.01):
                signal = "BUY"
        elif downtrend:
            # If the current price is near the 20 EMA => SELL
            if current_price > (ema_20 * 0.99):
                signal = "SELL"
        else:
            signal = "HOLD"

        # 4. Determine confidence
        # This is entirely optional. You can define any scoring logic you like
        confidence = 0.5  # default
        if uptrend and signal == "BUY":
            confidence = self.CONFIDENCE_THRESHOLD_UPTREND
        elif downtrend and signal == "SELL":
            confidence = self.CONFIDENCE_THRESHOLD_DOWNTREND
        else:
            confidence = 0.3  # lower confidence in sideway

        # 5. Calculate Stop Loss and Take Profit using ATR or a simple offset
        stop_loss, take_profit = self.calculate_stop_take(signal, current_price, atr)

        # 6. Position sizing
        position_size = self.dynamic_position_sizing(account_balance, confidence)

        # Build final dictionary
        result = {
            "signal": signal,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence": confidence,
            "position_size": position_size,
            "max_holding_time": 60,     # e.g. 60 seconds or any logic you prefer
            "trailing_stop": 0.0       # you can define a dynamic trailing stop if needed
        }
        if self.logger:
            self.logger.info(f"Rule-based signal: {result}")
        return result

    def compute_ema(self, prices: List[float], window: int):
        if len(prices) < window:
            return None
        multiplier = 2 / (window + 1)
        # Seed initial EMA with the first price or a simple SMA
        ema_val = sum(prices[:window]) / window
        for p in prices[window:]:
            ema_val = (p - ema_val) * multiplier + ema_val
        return ema_val

    def compute_atr(self, prices: List[float], window: int = 14):
        """
        A simplified ATR approach for demonstration.
        For actual ATR, you'd need high/low/close data. Here we approximate with close changes.
        """
        if len(prices) < window + 1:
            return 0.0
        ranges = []
        for i in range(1, len(prices)):
            ranges.append(abs(prices[i] - prices[i-1]))
        # Average the last 'window' differences
        recent_ranges = ranges[-window:]
        return sum(recent_ranges) / len(recent_ranges) if recent_ranges else 0.0

    def calculate_stop_take(self, signal: str, current_price: float, atr: float):
        if atr < self.MIN_ATR:
            atr = self.MIN_ATR  # guard small ATR

        if signal == "BUY":
            stop_loss = current_price - (self.RISK_MULTIPLIER_SL * atr)
            take_profit = current_price + (self.RISK_MULTIPLIER_TP * atr)
        elif signal == "SELL":
            stop_loss = current_price + (self.RISK_MULTIPLIER_SL * atr)
            take_profit = current_price - (self.RISK_MULTIPLIER_TP * atr)
        else:
            # If HOLD, just set both to current price as placeholders
            stop_loss = current_price
            take_profit = current_price
        return stop_loss, take_profit

    def dynamic_position_sizing(self, account_balance: float, confidence: float):
        """
        For demonstration, let's say we stake 1% of account balance times confidence.
        E.g., if confidence=1.0 and balance=1000 => position_size=10
        """
        base_size = account_balance * 0.01
        return max(1, base_size * confidence)

    def default_hold_signal(self):
        return {
            "signal": "HOLD",
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "confidence": 0.0,
            "position_size": 0.0,
            "max_holding_time": 60,
            "trailing_stop": 0.0
        }
