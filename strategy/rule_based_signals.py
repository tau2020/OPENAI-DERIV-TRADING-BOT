# strategy/rule_based_signals.py

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
        A scalping approach that tries to catch small moves in the short-term trend direction.
        - We compute a short EMA(3) and a slightly slower EMA(8).
        - If EMA(3) > EMA(8), we BUY with small SL/TP.
        - If EMA(3) < EMA(8), we SELL with small SL/TP.
        - If they’re nearly equal, we HOLD (avoid a choppy sideways state).
        
        This is purely an example. Adjust the offsets, windows, or add more conditions if needed.
        """

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
            # EMAs basically equal => sideways => skip
            signal = "HOLD"
        elif fast_ema > slow_ema:
            signal = "BUY"
        else:
            signal = "SELL"

        # 4) If HOLD, just return
        if signal == "HOLD":
            return self.default_hold_signal()

        # 5) Define small stop-loss & take-profit around current_close
        #    For a BUY, SL = current_close - SMALL_SL, TP = current_close + SMALL_TP
        #    For a SELL, SL = current_close + SMALL_SL, TP = current_close - SMALL_TP
        if signal == "BUY":
            stop_loss = current_close - self.SMALL_SL
            take_profit = current_close + self.SMALL_TP
        else:  # SELL
            stop_loss = current_close + self.SMALL_SL
            take_profit = current_close - self.SMALL_TP

        # 6) Confidence = 1.0 for frequent trading
        #    So the main strategy won't filter us out if it has a confidence threshold
        confidence = 1.0

        # 7) Position size: small fraction (like 1%) of account for frequent trades
        position_size = account_balance * 0.01
        if position_size < 1.0:
            position_size = 1.0  # or whatever minimal stake your broker requires

        # 8) Build final result
        result = {
            "signal": signal,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence": confidence,
            "position_size": position_size,
            "max_holding_time": self.SHORT_MAX_HOLDING_TIME,
            "trailing_stop": 0.0  # optional, you can implement a small trailing stop if you like
        }

        if self.logger:
            self.logger.info(f"Scalping-based rule signal: {result}")
        return result

    #
    # HELPER METHODS
    #

    def compute_ema(self, data: List[float], window: int):
        """
        Basic EMA calculation over entire series, returning the latest EMA value.
        """
        if len(data) < window:
            return None
        multiplier = 2 / (window + 1)
        ema_val = sum(data[:window]) / window  # start with SMA for first "window"
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
