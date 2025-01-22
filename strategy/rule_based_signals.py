#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
Title:     Enhanced Micro-Scalper with ATR/Volume Filters & Confidence Threshold
Author:    Neo (THE ONE)
Version:   3.2 (with extra logging at each filter step)
================================================================================
Description:
    This module demonstrates an enhanced version of the micro-scalper approach,
    incorporating additional filters to reduce whipsaw trades and improve 
    overall entry quality. Now with extra info logs after each filter, so we 
    can see which filter(s) cause the signal to remain in HOLD.

    1) **Confidence Threshold**:
       - Only take signals when confidence >= self.MIN_CONFIDENCE

    2) **ATR Filter** (volatility check):
       - Compute a short ATR. If ATR < self.MIN_ATR, skip trades.

    3) **Volume Filter**:
       - If average volume over the last N candles < self.MIN_VOLUME, skip trades.

    4) **Multi-Candle Momentum**:
       - Ensure the last M candles each exceed a minimum momentum threshold 
         (e.g., 0.0005 relative change). If not, skip trades.

    5) **(Optional) Time-of-Day Filter**: (not included here)
================================================================================
"""
from typing import List
import numpy as np

class RuleBasedSignals:
    def __init__(self, logger=None):
        self.logger = logger

        # Very short windows for immediate momentum
        self.VERY_FAST_EMA_WINDOW = 9
        self.FAST_EMA_WINDOW      = 10
        self.MICRO_WINDOW         = 15

        # Probability threshold for bullish/bearish distribution
        self.MIN_PROB_THRESHOLD   = 0.55

        # Price-based scalping distances
        self.DEFAULT_STOP_LOSS_POINTS   = 10.0
        self.DEFAULT_TAKE_PROFIT_POINTS = 5.0
        self.DEFAULT_TRAILING_STOP      = 4.0

        # Confidence threshold
        self.MIN_CONFIDENCE       = 0.30

        # ATR-based volatility filter
        self.ATR_WINDOW           = 7
        self.MIN_ATR              = 0.2

        # Volume filter
        self.MIN_VOLUME           = 0.5
        self.VOLUME_LOOKBACK      = 5

        # Multi-candle momentum check
        self.MIN_MOMENTUM_PCT     = 0.0001
        self.MOMENTUM_LOOKBACK    = 2

        # Risk-based position sizing
        self.RISK_PCT             = 0.01
        self.MAX_POSITION_SIZE    = 50.0

        # Hard-coded max holding time
        self.MAX_HOLDING_TIME     = 3

    def generate_candle_signal(
        self,
        candle_times: List,
        candle_opens: List[float],
        candle_highs: List[float],
        candle_lows: List[float],
        candle_closes: List[float],
        candle_volumes: List[float],
        account_balance: float = 1000.0
    ) -> dict:

        n = len(candle_closes)
        min_required = max(self.MICRO_WINDOW, self.VERY_FAST_EMA_WINDOW,
                           self.FAST_EMA_WINDOW, self.ATR_WINDOW)

        # 0) Check we have enough data
        if n < min_required:
            if self.logger:
                self.logger.debug(
                    f"[FilterFail] Not enough candles (have={n}, need={min_required}). => HOLD."
                )
            return self.default_hold_signal()
        if self.logger:
            self.logger.info(f"[FilterCheck] Candle count={n}, min_required={min_required}")

        # A) Micro distribution check => prob_bull
        bullish_count = 0
        for i in range(n - self.MICRO_WINDOW, n):
            if candle_closes[i] > candle_opens[i]:
                bullish_count += 1
        micro_bullish_prob = bullish_count / self.MICRO_WINDOW

        if self.logger:
            self.logger.info(
                f"[FilterCheck] micro_bullish_prob={micro_bullish_prob:.3f} "
                f"(threshold={self.MIN_PROB_THRESHOLD:.3f})"
            )

        # B) Very short EMA cross => immediate slope
        very_fast_ema = self.compute_ema(candle_closes, self.VERY_FAST_EMA_WINDOW)
        fast_ema      = self.compute_ema(candle_closes, self.FAST_EMA_WINDOW)
        if very_fast_ema is None or fast_ema is None:
            if self.logger:
                self.logger.debug("[FilterFail] EMA could not be computed => HOLD.")
            return self.default_hold_signal()

        if self.logger:
            self.logger.info(
                f"[FilterCheck] EMA windows => very_fast_ema={very_fast_ema:.5f}, "
                f"fast_ema={fast_ema:.5f}"
            )

        short_trend_signal = "HOLD"
        if very_fast_ema > fast_ema:
            short_trend_signal = "BUY"
        elif very_fast_ema < fast_ema:
            short_trend_signal = "SELL"

        if self.logger:
            self.logger.info(
                f"[FilterCheck] short_trend_signal={short_trend_signal}, "
                f"very_fast_ema={very_fast_ema:.5f}, fast_ema={fast_ema:.5f}"
            )

        # Combine micro_bullish_prob & short_trend_signal
        signal = "HOLD"
        # If bullish prob >= threshold & short trend=BUY => BUY
        # If (1-bullish prob) >= threshold & short trend=SELL => SELL
        if micro_bullish_prob >= self.MIN_PROB_THRESHOLD and short_trend_signal == "BUY":
            signal = "BUY"
        elif (1.0 - micro_bullish_prob) >= self.MIN_PROB_THRESHOLD and short_trend_signal == "SELL":
            signal = "SELL"

        if signal == "HOLD":
            if self.logger:
                self.logger.info(
                    f"[FilterFail] micro_bullish_prob={micro_bullish_prob:.3f} "
                    f"vs threshold={self.MIN_PROB_THRESHOLD:.3f}, short_trend_signal={short_trend_signal} => HOLD."
                )
            return self.default_hold_signal()

        # C) ATR filter => skip if too quiet
        current_atr = self.compute_atr(candle_highs, candle_lows, candle_closes, self.ATR_WINDOW)
        if current_atr is None:
            if self.logger:
                self.logger.debug("[FilterFail] ATR None => HOLD.")
            return self.default_hold_signal()
        if self.logger:
            self.logger.info(
                f"[FilterCheck] current_atr={current_atr:.3f} (threshold={self.MIN_ATR:.3f})"
            )
        if current_atr < self.MIN_ATR:
            if self.logger:
                self.logger.info(
                    f"[FilterFail] ATR={current_atr:.3f} < MIN_ATR={self.MIN_ATR:.3f} => HOLD."
                )
            return self.default_hold_signal()

        # D) Volume filter
        volume_lookback_slice = candle_volumes[-self.VOLUME_LOOKBACK:]
        avg_vol = float(np.mean(volume_lookback_slice)) if volume_lookback_slice else 0.0
        if self.logger:
            self.logger.info(
                f"[FilterCheck] avg_vol={avg_vol:.3f} (threshold={self.MIN_VOLUME:.3f}), "
                f"volume_lookback={volume_lookback_slice}"
            )
        if avg_vol < self.MIN_VOLUME:
            if self.logger:
                self.logger.info(
                    f"[FilterFail] AvgVol={avg_vol:.3f} < MIN_VOLUME={self.MIN_VOLUME:.3f} => HOLD."
                )
            return self.default_hold_signal()

        # E) Multi-candle momentum check
        """ if not self.check_momentum(candle_opens, candle_closes, lookback=self.MOMENTUM_LOOKBACK):
            if self.logger:
                self.logger.info(
                    f"[FilterFail] Multi-candle momentum < {self.MIN_MOMENTUM_PCT:.4f} => HOLD."
                )
            return self.default_hold_signal()
        else:
            if self.logger:
                self.logger.info(
                    f"[FilterCheck] Momentum check PASSED => last {self.MOMENTUM_LOOKBACK} candles >= {self.MIN_MOMENTUM_PCT:.4f}"
                ) """

        # F) Confidence
        last_close = candle_closes[-1]
        spread_mag = abs(very_fast_ema - fast_ema)
        spread_factor = min(spread_mag / max(last_close, 1e-6), 1.0)
        if signal == "BUY":
            raw_confidence = micro_bullish_prob
        else:
            raw_confidence = (1.0 - micro_bullish_prob)
        confidence = raw_confidence * (0.5 + 0.5 * spread_factor)
        confidence = min(1.0, max(0.0, confidence))

        if self.logger:
            self.logger.info(
                f"[FilterCheck] confidence={confidence:.3f}, raw_confidence={raw_confidence:.3f}, "
                f"spread_factor={spread_factor:.3f}, MIN_CONFIDENCE={self.MIN_CONFIDENCE:.3f}"
            )

        if confidence < self.MIN_CONFIDENCE:
            if self.logger:
                self.logger.info(
                    f"[FilterFail] confidence={confidence:.3f} < MIN_CONFIDENCE={self.MIN_CONFIDENCE:.3f} => HOLD."
                )
            return self.default_hold_signal()

        # G) Final check => build signal
        stop_loss_points     = self.DEFAULT_STOP_LOSS_POINTS
        take_profit_points   = self.DEFAULT_TAKE_PROFIT_POINTS
        trailing_stop_points = self.DEFAULT_TRAILING_STOP

        position_size = (account_balance * self.RISK_PCT) / max(stop_loss_points, 1e-6)
        position_size = min(position_size, self.MAX_POSITION_SIZE)

        result = {
            "signal": signal,
            "stop_loss_points": stop_loss_points,
            "take_profit_points": take_profit_points,
            "trailing_stop_points": trailing_stop_points,
            "confidence": confidence,
            "position_size": position_size,
            "max_holding_time": self.MAX_HOLDING_TIME
        }
        if self.logger:
            self.logger.info(
                f"[EnhancedMicroScalper] => direction={signal}, prob(bull)={micro_bullish_prob:.3f}, "
                f"emaVF={very_fast_ema:.5f}, emaF={fast_ema:.5f}, conf={confidence:.3f}, "
                f"ATR={current_atr:.3f}, avgVol={avg_vol:.3f}, SL={stop_loss_points:.2f}, "
                f"TP={take_profit_points:.2f}, trail={trailing_stop_points:.2f}, stake={position_size:.2f}"
            )
        return result

    def compute_ema(self, data: List[float], window: int) -> float:
        if len(data) < window:
            return None
        multiplier = 2 / (window + 1)
        ema_val = np.mean(data[:window])  # initial average
        for price in data[window:]:
            ema_val = (price - ema_val) * multiplier + ema_val
        return ema_val

    def compute_atr(self, highs, lows, closes, window):
        """
        Standard ATR over 'window' periods, returning the most recent ATR value.
        """
        n = len(closes)
        if n < window + 1:
            return None
        trs = []
        for i in range(n - window, n):
            current_high = highs[i]
            current_low  = lows[i]
            prev_close   = closes[i - 1]
            tr1 = current_high - current_low
            tr2 = abs(current_high - prev_close)
            tr3 = abs(current_low - prev_close)
            trs.append(max(tr1, tr2, tr3))
        return float(np.mean(trs))

    def check_momentum(self, opens, closes, lookback=2) -> bool:
        """
        Ensures the last 'lookback' candles each have at least self.MIN_MOMENTUM_PCT
        of price change from open to close. If not, return False.
        """
        if len(closes) < lookback:
            return False
        recent_opens  = opens[-lookback:]
        recent_closes = closes[-lookback:]
        for o, c in zip(recent_opens, recent_closes):
            if o == 0:
                return False
            momentum_pct = abs((c - o) / o)
            if momentum_pct < self.MIN_MOMENTUM_PCT:
                return False
        return True

    def default_hold_signal(self) -> dict:
        return {
            "signal": "HOLD",
            "stop_loss_points": 0.0,
            "take_profit_points": 0.0,
            "trailing_stop_points": 0.0,
            "confidence": 0.0,
            "position_size": 0.0,
            "max_holding_time": 1
        }
