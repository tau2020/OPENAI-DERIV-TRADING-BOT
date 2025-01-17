from typing import List
import numpy as np

class RuleBasedSignals:
    def __init__(self, logger=None):
        self.logger = logger

        #
        # Scalping thresholds / tunable parameters specifically for R_10
        #
        
        # 1. Adjusted SL/TP for improved reward-to-risk
        #    - Stop ~0.0002 away, Take-profit ~0.0007 => R:R ~3.5:1
        self.SCALP_SL = 0.0002   # e.g., stops ~0.0002 points away
        self.SCALP_TP = 0.0007   # e.g., take-profit ~0.0007 points away

        # Because R_10 often moves slowly, we reduce this threshold slightly.
        # If the difference between FAST_EMA and SLOW_EMA is under this value,
        # we consider it “no trend”.
        self.EMA_SPREAD_THRESHOLD = 8e-7  # lowered from 1e-6 => more sensitivity

        # 2. Momentum check
        #    The last candle move (close - open)/open must exceed this threshold
        #    to confirm a bullish or bearish push.
        self.MOMENTUM_THRESHOLD = 0.0002

        # 3. Shorter windows can catch micro-trends quickly;
        #    reduced SLOW_EMA_WINDOW from 5 to 4 for slightly faster reaction
        self.FAST_EMA_WINDOW = 2
        self.SLOW_EMA_WINDOW = 4

        # 4. Max holding time (in minutes or candle intervals)
        self.SHORT_MAX_HOLDING_TIME = 2

        # 5. Candle-range filtering
        self.CANDLE_RANGE_LOOKBACK = 5
        self.MIN_RANGE_FACTOR = 1.2

        # 6. Optional trailing stop if momentum is strong
        self.TRAILING_STOP_DISTANCE = 0.00015  # adjusts dynamically based on momentum

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
        Rule-based scalping signals for Deriv Synthetic Index 10 (R_10).
        - Quickly identifies momentum-based micro-trends with short EMAs (2,4).
        - Imposes a minimum spread requirement (EMA_SPREAD_THRESHOLD).
        - Uses candle-range filtering to avoid entering during tiny/flat candles.
        - Returns small SL/TP, suitable for short, frequent trades.
        - OPTIONAL: dynamic trailing stop if momentum is strongly above threshold.
        """

        # 1) Ensure enough candles to compute the SLOW EMA
        if len(candle_closes) < self.SLOW_EMA_WINDOW:
            if self.logger:
                self.logger.debug("Not enough candle data for EMAs. Returning HOLD.")
            return self.default_hold_signal()

        # 2) Compute the short EMAs
        fast_ema = self.compute_ema(candle_closes, self.FAST_EMA_WINDOW)
        slow_ema = self.compute_ema(candle_closes, self.SLOW_EMA_WINDOW)
        if fast_ema is None or slow_ema is None:
            if self.logger:
                self.logger.debug("Unable to compute EMAs. Returning HOLD.")
            return self.default_hold_signal()

        ema_spread = fast_ema - slow_ema
        if abs(ema_spread) < self.EMA_SPREAD_THRESHOLD:
            # EMAs are too close => no strong trend => HOLD
            if self.logger:
                self.logger.debug(
                    f"EMA spread ({ema_spread:.7f}) < threshold => likely sideways => HOLD."
                )
            return self.default_hold_signal()

        # 3) Momentum check (last candle only)
        last_close = candle_closes[-1]
        last_open = candle_opens[-1]
        if last_open == 0.0:
            if self.logger:
                self.logger.debug("Last candle open is 0.0 => Invalid data => HOLD.")
            return self.default_hold_signal()

        last_candle_move_pct = (last_close - last_open) / last_open

        # OPTIONAL: multi-candle momentum check (slightly stricter)
        # If you want to confirm the last two candles were moving in the same direction:
        # if len(candle_closes) >= 2:
        #     second_last_close = candle_closes[-2]
        #     second_last_open = candle_opens[-2]
        #     second_last_move_pct = (second_last_close - second_last_open) / second_last_open
        #     # Check if both last candle & second last candle have same sign of momentum
        #     if not (last_candle_move_pct * second_last_move_pct > 0 and
        #             abs(last_candle_move_pct) > self.MOMENTUM_THRESHOLD and
        #             abs(second_last_move_pct) > self.MOMENTUM_THRESHOLD):
        #         # If you want BOTH candles to confirm, you'd return HOLD here.
        #         return self.default_hold_signal()

        # 4) Candle range filter
        last_candle_range = candle_highs[-1] - candle_lows[-1]
        start_index = max(0, len(candle_closes) - self.CANDLE_RANGE_LOOKBACK)
        recent_ranges = [
            (candle_highs[i] - candle_lows[i])
            for i in range(start_index, len(candle_closes))
        ]
        avg_recent_range = np.mean(recent_ranges) if len(recent_ranges) > 0 else 0.0

        if avg_recent_range > 0 and last_candle_range < self.MIN_RANGE_FACTOR * avg_recent_range:
            # Candle not volatile enough => HOLD
            if self.logger:
                self.logger.debug(
                    f"Last candle range ({last_candle_range:.5f}) < "
                    f"{self.MIN_RANGE_FACTOR}x average range ({avg_recent_range:.5f}) => HOLD."
                )
            return self.default_hold_signal()

        # 5) Derive signal direction from EMAs + last candle momentum
        if ema_spread > 0 and last_candle_move_pct > self.MOMENTUM_THRESHOLD:
            signal = "BUY"
        elif ema_spread < 0 and last_candle_move_pct < -self.MOMENTUM_THRESHOLD:
            signal = "SELL"
        else:
            signal = "HOLD"

        if signal == "HOLD":
            return self.default_hold_signal()

        # 6) Define small scalping stop-loss & take-profit
        if signal == "BUY":
            stop_loss = last_close - self.SCALP_SL
            take_profit = last_close + self.SCALP_TP
        else:  # SELL
            stop_loss = last_close + self.SCALP_SL
            take_profit = last_close - self.SCALP_TP

        # 7) Confidence: set to 1.0 to ensure the main strategy layer doesn't ignore it
        confidence = 1.0

        # 8) Position sizing: for a small balance, stake might be 1%
        position_size = account_balance * 0.01
        if position_size < 1.0:
            position_size = 1.0  # or the broker's actual minimum stake

        # 9) OPTIONAL: dynamic trailing stop if momentum is significantly above the threshold
        trailing_stop = 0.0
        if abs(last_candle_move_pct) > self.MOMENTUM_THRESHOLD * 2:
            # If momentum is quite strong, allow a trailing stop
            if signal == "BUY":
                trailing_stop = self.TRAILING_STOP_DISTANCE
            else:
                trailing_stop = self.TRAILING_STOP_DISTANCE
            if self.logger:
                self.logger.debug(
                    f"Strong momentum => applying trailing stop of {trailing_stop:.7f}"
                )

        # Build final result
        result = {
            "signal": signal,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence": confidence,
            "position_size": position_size,
            "max_holding_time": self.SHORT_MAX_HOLDING_TIME,
            "trailing_stop": trailing_stop
        }

        if self.logger:
            self.logger.info(f"[R_10 Scalping] Generated signal: {result}")
        return result

    #
    # HELPER METHODS
    #
    def compute_ema(self, data: List[float], window: int) -> float:
        """
        Basic EMA calculation over the entire series, returning ONLY the latest EMA value.
        """
        if len(data) < window:
            return None

        multiplier = 2 / (window + 1)
        # Start with a simple moving average for the first 'window' data points
        ema_val = sum(data[:window]) / window

        # Then apply the EMA formula over the rest
        for price in data[window:]:
            ema_val = (price - ema_val) * multiplier + ema_val

        return ema_val

    def default_hold_signal(self) -> dict:
        """
        Returns a default 'do nothing' signal with minimal placeholders.
        """
        return {
            "signal": "HOLD",
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "confidence": 0.0,
            "position_size": 0.0,
            "max_holding_time": 1,
            "trailing_stop": 0.0
        }
