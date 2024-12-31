# strategy/rule_based_signals.py

import statistics
from typing import List
from utils.analysis_utils import analyze_volume

class RuleBasedSignals:
    def __init__(self, logger=None):
        self.logger = logger
        self.MIN_ATR = 0.0001
        self.RISK_MULTIPLIER_SL = 1.5
        self.RISK_MULTIPLIER_TP = 2.0
        self.CONFIDENCE_THRESHOLD_UPTREND = 0.8
        self.CONFIDENCE_THRESHOLD_DOWNTREND = 0.8

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
        """
        Use O/H/L/C volume for more robust signals.
        """
        if len(candle_closes) < 20:
            if self.logger:
                self.logger.debug("Not enough candles to compute signals. Returning HOLD.")
            return self.default_hold_signal()

        current_close = candle_closes[-1]
        atr = self.compute_atr(candle_highs, candle_lows, candle_closes, window=14)
        volume_ratio = analyze_volume(candle_volumes, lookback=5) if candle_volumes else 1.0

        ema_20 = self.compute_ema(candle_closes, 20)
        ema_50 = self.compute_ema(candle_closes, 50)
        ema_200 = self.compute_ema(candle_closes, 200)

        uptrend = False
        downtrend = False
        if ema_20 and ema_50 and ema_200:
            uptrend = (ema_20 > ema_50 > ema_200)
            downtrend = (ema_20 < ema_50 < ema_200)

        signal = "HOLD"
        if uptrend and volume_ratio > 1.5:
            if current_close <= (ema_20 * 1.01):
                signal = "BUY"
        elif downtrend and volume_ratio > 1.5:
            if current_close >= (ema_20 * 0.99):
                signal = "SELL"

        if signal == "BUY" and uptrend:
            confidence = self.CONFIDENCE_THRESHOLD_UPTREND
        elif signal == "SELL" and downtrend:
            confidence = self.CONFIDENCE_THRESHOLD_DOWNTREND
        else:
            confidence = 0.3

        if signal in ["BUY", "SELL"]:
            stop_loss, take_profit = self.calculate_stop_take(signal, current_close, atr)
        else:
            stop_loss = 0.0
            take_profit = 0.0

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
            self.logger.info(f"Candle-based rule signal: {result}")
        return result

    def compute_ema(self, data: List[float], window: int):
        if len(data) < window:
            return None
        multiplier = 2 / (window + 1)
        ema_val = sum(data[:window]) / window
        for p in data[window:]:
            ema_val = (p - ema_val) * multiplier + ema_val
        return ema_val

    def compute_atr(self, highs: List[float], lows: List[float], closes: List[float], window: int = 14):
        """
        True Range often uses: max(high-low, abs(high-prevClose), abs(low-prevClose)).
        For simplicity, you can approximate. Here we'll do a basic approach.
        """
        if len(highs) < window + 1:
            return 0.0
        true_ranges = []
        for i in range(1, len(highs)):
            hl_range = abs(highs[i] - lows[i])
            cp_range1 = abs(highs[i] - closes[i - 1])
            cp_range2 = abs(lows[i] - closes[i - 1])
            tr = max(hl_range, cp_range1, cp_range2)
            true_ranges.append(tr)

        recent_tr = true_ranges[-window:]
        if not recent_tr:
            return self.MIN_ATR
        avg_range = sum(recent_tr) / len(recent_tr)
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
