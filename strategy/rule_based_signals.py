# strategy/rule_based_signals.py

import statistics
from typing import List
from utils.analysis_utils import analyze_volume

class RuleBasedSignals:
    """
    A simple rule-based signal generator that calculates indicators and
    produces signals (BUY, SELL, HOLD) along with SL/TP.
    """

    def __init__(self, logger=None):
        self.logger = logger
        self.MIN_ATR = 0.0001
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
        recent_trades_win_loss_ratio: float = 1.0,
        account_balance: float = 1000.0,
    ):
        if not prices or len(prices) < 20:
            if self.logger:
                self.logger.debug("Not enough prices to compute EMAs. Returning HOLD.")
            return self.default_hold_signal()

        current_price = prices[-1]
        ema_20 = self.compute_ema(prices, 20)
        ema_50 = self.compute_ema(prices, 50)
        ema_200 = self.compute_ema(prices, 200)
        atr = self.compute_atr(prices, window=14)

        uptrend = False
        downtrend = False
        if ema_20 and ema_50 and ema_200:
            uptrend = (ema_20 > ema_50 > ema_200)
            downtrend = (ema_20 < ema_50 < ema_200)

        # Analyze volume ratio
        volume_ratio = 1.0
        if volume_data:
            volume_ratio = analyze_volume(volume_data, lookback=5)
        signal = "HOLD"

        if uptrend and volume_ratio > 1.5:
            if current_price <= ema_20 * 1.01:
                signal = "BUY"
        elif downtrend and volume_ratio > 1.5:
            if current_price >= ema_20 * 0.99:
                signal = "SELL"

        if signal == "BUY" and uptrend:
            confidence = self.CONFIDENCE_THRESHOLD_UPTREND
        elif signal == "SELL" and downtrend:
            confidence = self.CONFIDENCE_THRESHOLD_DOWNTREND
        else:
            confidence = 0.3

        stop_loss, take_profit = self.calculate_stop_take(signal, current_price, atr)
        position_size = self.dynamic_position_sizing(account_balance, confidence)

        result = {
            "signal": signal,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence": confidence,
            "position_size": position_size,
            "max_holding_time": 60,
            "trailing_stop": 0.0
        }

        if self.logger:
            self.logger.info(f"Rule-based signal: {result}")
        return result

    def compute_ema(self, prices: List[float], window: int):
        if len(prices) < window:
            return None
        multiplier = 2 / (window + 1)
        ema_val = sum(prices[:window]) / window
        for p in prices[window:]:
            ema_val = (p - ema_val) * multiplier + ema_val
        return ema_val

    def compute_atr(self, prices: List[float], window: int = 14):
        if len(prices) < window + 1:
            return 0.0
        changes = []
        for i in range(1, len(prices)):
            changes.append(abs(prices[i] - prices[i - 1]))
        recent_changes = changes[-window:]
        avg_range = sum(recent_changes) / len(recent_changes) if recent_changes else 0.0
        return max(avg_range, self.MIN_ATR)

    def calculate_stop_take(self, signal: str, current_price: float, atr: float):
        if signal == "BUY":
            stop_loss = current_price - (self.RISK_MULTIPLIER_SL * atr)
            take_profit = current_price + (self.RISK_MULTIPLIER_TP * atr)
        elif signal == "SELL":
            stop_loss = current_price + (self.RISK_MULTIPLIER_SL * atr)
            take_profit = current_price - (self.RISK_MULTIPLIER_TP * atr)
        else:
            stop_loss = current_price
            take_profit = current_price
        return stop_loss, take_profit

    def dynamic_position_sizing(self, account_balance: float, confidence: float):
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
