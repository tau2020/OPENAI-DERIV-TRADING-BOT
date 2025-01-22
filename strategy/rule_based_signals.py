#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
Title:     Enhanced Micro-Scalper (Forex) - Extended Filters
Author:    Neo (THE ONE)
Version:   4.0
================================================================================
Description:
    An advanced micro-scalper approach for a typical Forex pair (e.g. AUD/JPY),
    with multiple filters to improve entry accuracy:

    1) Very short EMA cross + micro distribution (existing).
    2) ATR filter to avoid low-volatility phases.
    3) Volume filter (if you have meaningful volume data).
    4) RSI filter: only buy if RSI < 70, only sell if RSI > 30, etc.
    5) Time-of-Day filter: trade only within an allowed window (e.g. 07:00-17:00).
    6) Multi-candle momentum check: skip if the last N candles had minimal change.
    7) (Optional) Basic candlestick pattern check, e.g. skip if you see doji or
       some pattern you don't like.

Adjust constants to suit your strategy, time frames, risk tolerance, etc.
================================================================================
"""
from typing import List
import numpy as np
from datetime import time

class RuleBasedSignals:
    def __init__(self, logger=None):
        self.logger = logger

        # Very short EMA windows for immediate momentum
        self.VERY_FAST_EMA_WINDOW = 9
        self.FAST_EMA_WINDOW      = 10
        self.MICRO_WINDOW         = 15

        # Probability threshold for bullish/bearish distribution
        self.MIN_PROB_THRESHOLD   = 0.55

        # Price-based scalping distances (in typical 'price' terms for e.g. AUD/JPY)
        self.DEFAULT_STOP_LOSS_POINTS   = 0.20  # ~20 pips if 1 pip=0.01
        self.DEFAULT_TAKE_PROFIT_POINTS = 0.30  # ~30 pips
        self.DEFAULT_TRAILING_STOP      = 0.15  # ~15 pips

        # Confidence threshold
        self.MIN_CONFIDENCE = 0.30

        # ATR-based volatility filter
        self.ATR_WINDOW = 7
        self.MIN_ATR    = 0.05   # skip if ATR < 5 pips

        # Volume filter
        self.MIN_VOLUME      = 1.0
        self.VOLUME_LOOKBACK = 5

        # RSI filter
        self.RSI_WINDOW     = 14
        self.RSI_BUY_LEVEL  = 70  # won't BUY if RSI >= 70 (overbought)
        self.RSI_SELL_LEVEL = 30  # won't SELL if RSI <= 30 (oversold)

        # Multi-candle momentum check
        self.MIN_MOMENTUM_PCT  = 0.0001
        self.MOMENTUM_LOOKBACK = 2
        self.ENABLE_MOMENTUM_CHECK = True

        # Risk-based position sizing
        self.RISK_PCT          = 0.01
        self.MAX_POSITION_SIZE = 2.0

        # Hard-coded max holding time
        self.MAX_HOLDING_TIME  = 3

        # Time-of-Day filter: only trade between these times (in "local" or "UTC" candle times)
        # Example: start=7:00, end=17:00
        self.ALLOWED_START_TIME = time(23, 30)
        self.ALLOWED_END_TIME   = time(23, 50)
        self.ENABLE_TIME_FILTER = True

        # Candlestick pattern filter (placeholder, can add your own logic)
        self.ENABLE_PATTERN_CHECK = False

    def generate_candle_signal(
        self,
        candle_times:   List,    # list[datetime]
        candle_opens:   List[float],
        candle_highs:   List[float],
        candle_lows:    List[float],
        candle_closes:  List[float],
        candle_volumes: List[float],
        account_balance: float = 1000.0
    ) -> dict:
        """
        Main method to produce a 'BUY', 'SELL', or 'HOLD' signal using:
          - short-term EMA crossover
          - micro distribution
          - optional RSI filter
          - ATR filter
          - volume filter
          - time-of-day filter
          - multi-candle momentum
          - optional candlestick pattern filter
        """
        n = len(candle_closes)
        min_required = max(
            self.MICRO_WINDOW,
            self.VERY_FAST_EMA_WINDOW,
            self.FAST_EMA_WINDOW,
            self.ATR_WINDOW,
            self.RSI_WINDOW  # we also need enough data to compute RSI
        )

        # 0) Check we have enough data
        if n < min_required:
            if self.logger:
                self.logger.debug(
                    f"[FilterFail] Not enough candles (have={n}, need={min_required}). => HOLD."
                )
            return self.default_hold_signal()

        if self.logger:
            self.logger.info(f"[FilterCheck] Candle count={n}, min_required={min_required}")

        # (Optional) Time-of-Day Filter
        if self.ENABLE_TIME_FILTER:
            if not candle_times:
                if self.logger:
                    self.logger.debug("[FilterFail] No candle_times data => HOLD.")
                return self.default_hold_signal()

            last_time = candle_times[-1].time()
            if not (self.ALLOWED_START_TIME <= last_time <= self.ALLOWED_END_TIME):
                if self.logger:
                    self.logger.info(
                        f"[FilterFail] Outside allowed trading window {self.ALLOWED_START_TIME}-{self.ALLOWED_END_TIME}: {last_time} => HOLD."
                    )
                return self.default_hold_signal()

        # A) Micro distribution check => prob_bull
        micro_bullish_prob = self.compute_micro_distribution(
            candle_opens, candle_closes, self.MICRO_WINDOW
        )

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
                f"[FilterCheck] EMA => vfast={very_fast_ema:.5f}, fast={fast_ema:.5f}"
            )

        short_trend_signal = "HOLD"
        if very_fast_ema > fast_ema:
            short_trend_signal = "BUY"
        elif very_fast_ema < fast_ema:
            short_trend_signal = "SELL"

        if self.logger:
            self.logger.info(
                f"[FilterCheck] short_trend_signal={short_trend_signal}, "
                f"emaVF={very_fast_ema:.5f}, emaF={fast_ema:.5f}"
            )

        # Combine micro_bullish_prob & short_trend_signal
        signal = "HOLD"
        if micro_bullish_prob >= self.MIN_PROB_THRESHOLD and short_trend_signal == "BUY":
            signal = "BUY"
        elif (1.0 - micro_bullish_prob) >= self.MIN_PROB_THRESHOLD and short_trend_signal == "SELL":
            signal = "SELL"

        if signal == "HOLD":
            if self.logger:
                self.logger.info(
                    f"[FilterFail] micro_bullish_prob={micro_bullish_prob:.3f} vs "
                    f"th={self.MIN_PROB_THRESHOLD:.3f}, short_trend_signal={short_trend_signal} => HOLD."
                )
            return self.default_hold_signal()

        # C) ATR filter => skip if too quiet
        current_atr = self.compute_atr(candle_highs, candle_lows, candle_closes, self.ATR_WINDOW)
        if not current_atr:
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
        avg_vol = self.compute_avg_volume(candle_volumes, self.VOLUME_LOOKBACK)
        if self.logger:
            self.logger.info(
                f"[FilterCheck] avg_vol={avg_vol:.3f} (threshold={self.MIN_VOLUME:.3f})"
            )

        if avg_vol < self.MIN_VOLUME:
            if self.logger:
                self.logger.info(
                    f"[FilterFail] AvgVol={avg_vol:.3f} < MIN_VOLUME={self.MIN_VOLUME:.3f} => HOLD."
                )
            return self.default_hold_signal()

        # E) RSI filter
        current_rsi = self.compute_rsi(candle_closes, self.RSI_WINDOW)
        if current_rsi is None:
            if self.logger:
                self.logger.debug("[FilterFail] RSI None => HOLD.")
            return self.default_hold_signal()

        if self.logger:
            self.logger.info(
                f"[FilterCheck] RSI={current_rsi:.1f}, RSI_BUY_Lv={self.RSI_BUY_LEVEL}, RSI_SELL_Lv={self.RSI_SELL_LEVEL}"
            )

        # e.g., only BUY if RSI < RSI_BUY_LEVEL, only SELL if RSI > RSI_SELL_LEVEL
        if signal == "BUY" and current_rsi >= self.RSI_BUY_LEVEL:
            if self.logger:
                self.logger.info(
                    f"[FilterFail] RSI={current_rsi:.1f} >= RSI_BUY_LEVEL={self.RSI_BUY_LEVEL} => No BUY => HOLD."
                )
            return self.default_hold_signal()

        if signal == "SELL" and current_rsi <= self.RSI_SELL_LEVEL:
            if self.logger:
                self.logger.info(
                    f"[FilterFail] RSI={current_rsi:.1f} <= RSI_SELL_LEVEL={self.RSI_SELL_LEVEL} => No SELL => HOLD."
                )
            return self.default_hold_signal()

        # F) Multi-candle momentum check
        if self.ENABLE_MOMENTUM_CHECK:
            if not self.check_momentum(
                candle_opens,
                candle_closes,
                lookback=self.MOMENTUM_LOOKBACK,
                min_pct=self.MIN_MOMENTUM_PCT
            ):
                if self.logger:
                    self.logger.info(
                        f"[FilterFail] Multi-candle momentum < {self.MIN_MOMENTUM_PCT:.4f} => HOLD."
                    )
                return self.default_hold_signal()
            else:
                if self.logger:
                    self.logger.info(
                        f"[FilterCheck] Momentum check PASSED => last {self.MOMENTUM_LOOKBACK} candles >= {self.MIN_MOMENTUM_PCT:.4f}"
                    )

        # G) (Optional) Candlestick pattern check
        if self.ENABLE_PATTERN_CHECK:
            # e.g. skip if a doji forms => abs(open-close) < some fraction of (high-low)
            if self.check_doji_pattern(candle_opens[-1], candle_closes[-1],
                                       candle_highs[-1], candle_lows[-1]):
                if self.logger:
                    self.logger.info("[FilterFail] Candle pattern => DOJI => HOLD.")
                return self.default_hold_signal()

        # H) Confidence calculation
        last_close   = candle_closes[-1]
        spread_mag   = abs(very_fast_ema - fast_ema)
        spread_factor= min(spread_mag / max(last_close, 1e-6), 1.0)
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

        # I) Final => build signal
        stop_loss_points     = self.DEFAULT_STOP_LOSS_POINTS
        take_profit_points   = self.DEFAULT_TAKE_PROFIT_POINTS
        trailing_stop_points = self.DEFAULT_TRAILING_STOP

        position_size = (account_balance * self.RISK_PCT) / max(stop_loss_points, 1e-6)
        position_size = min(position_size, self.MAX_POSITION_SIZE)

        result = {
            "signal":               signal,
            "stop_loss_points":     stop_loss_points,
            "take_profit_points":   take_profit_points,
            "trailing_stop_points": trailing_stop_points,
            "confidence":           confidence,
            "position_size":        position_size,
            "max_holding_time":     self.MAX_HOLDING_TIME
        }

        if self.logger:
            self.logger.info(
                f"[EnhancedMicroScalper-FX-ADV] => dir={signal}, RSI={current_rsi:.1f}, conf={confidence:.3f}, "
                f"ATR={current_atr:.3f}, avgVol={avg_vol:.3f}, SL={stop_loss_points:.2f}, TP={take_profit_points:.2f}, "
                f"trail={trailing_stop_points:.2f}, stake={position_size:.2f}"
            )

        return result

    # --------------------------------------------------------------------------
    #  Helper Methods
    # --------------------------------------------------------------------------

    def compute_micro_distribution(self, opens, closes, window):
        """
        Fraction of last 'window' candles that are bullish vs total.
        e.g. 0.7 means 70% of the last 'window' candles had close > open.
        """
        segment_opens  = opens[-window:]
        segment_closes = closes[-window:]
        bullish_count  = sum(1 for o, c in zip(segment_opens, segment_closes) if c > o)
        return bullish_count / window

    def compute_ema(self, data: List[float], window: int) -> float:
        """
        Standard EMA calculation for a given window.
        """
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

    def compute_avg_volume(self, volumes, lookback=5):
        """
        Returns average volume over the last 'lookback' bars.
        """
        if len(volumes) < lookback:
            return 0.0
        slice_vol = volumes[-lookback:]
        return float(np.mean(slice_vol))

    def compute_rsi(self, closes: List[float], window: int) -> float:
        """
        Standard RSI calculation. If not enough data, return None.
        RSI = 100 - 100/(1 + RS), where RS=avg_gain/avg_loss (smoothed)
        """
        if len(closes) < window + 1:
            return None
        # Price changes
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = []
        losses = []
        # Initialize first average gain/loss
        first_slice = deltas[:window]
        avg_gain = np.mean([x for x in first_slice if x > 0]) if any(x > 0 for x in first_slice) else 0.0
        avg_loss= abs(np.mean([x for x in first_slice if x < 0])) if any(x < 0 for x in first_slice) else 0.0

        # Start from 'window' onward
        idx = window
        for change in deltas[window:]:
            # new gain/loss
            gain = max(change, 0.0)
            loss = abs(min(change, 0.0))
            # smoothing
            avg_gain = (avg_gain * (window - 1) + gain) / window
            avg_loss = (avg_loss * (window - 1) + loss) / window
            idx += 1

        # Final RSI
        if avg_loss == 0:
            # if no losses => RSI=100
            rsi = 100.0
        else:
            rs  = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        return rsi

    def check_momentum(self, opens, closes, lookback=2, min_pct=0.0001) -> bool:
        """
        Ensures the last 'lookback' candles each have at least 'min_pct'
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
            if momentum_pct < min_pct:
                return False
        return True

    def check_doji_pattern(self, open_price, close_price, high_price, low_price, doji_ratio=0.1) -> bool:
        """
        Example method: 'Doji' if open~close relative to overall range.
        doji_ratio=0.1 => if abs(open-close) < doji_ratio*(high-low), treat as doji.
        """
        candle_range = (high_price - low_price)
        body = abs(close_price - open_price)
        if candle_range < 1e-6:
            return False  # avoid division by zero
        return (body < (doji_ratio * candle_range))

    def default_hold_signal(self) -> dict:
        """
        Return a default 'HOLD' signal structure.
        """
        return {
            "signal":               "HOLD",
            "stop_loss_points":     0.0,
            "take_profit_points":   0.0,
            "trailing_stop_points": 0.0,
            "confidence":           0.0,
            "position_size":        0.0,
            "max_holding_time":     1
        }
