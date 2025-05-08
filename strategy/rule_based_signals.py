import numpy as np
from typing import List, Dict
from collections import deque

class RuleBasedSignals:
    """
    Optimised Rule-Based Signal Engine
    ----
    • Vectorised indicator calculations (EMA, RSI, ATR)
    • Incremental caching to avoid redundant recomputation
    • Adaptive risk filters linked to ATR & volatility
    """

    # ------------------------------------------------------------------ #
    #  INITIALISATION
    # ------------------------------------------------------------------ #
    def __init__(self, logger=None):
        self.logger = logger

        # Tunable parameters
        self.base_sl                 = 0.0010
        self.base_tp                 = 0.0015
        self.volatility_window       = 20
        self.atr_window              = 14
        self.rsi_window              = 14
        self.macd_fast               = 24
        self.macd_slow               = 50
        self.macd_signal             = 9
        self.volume_spike_multiplier = 3.0
        self.min_confidence          = 0.80
        self.max_confidence          = 0.95          # slightly wider band
        self.risk_per_trade          = 0.02          # 2 %

        # Rolling buffers
        self.price_data   = deque(maxlen=600)        # store > max(window)
        self.volume_data  = deque(maxlen=600)
        self.trade_history = deque(maxlen=100)

        # Indicator cache (length-keyed)
        self._cache: Dict[str, Dict[int, np.ndarray]] = {
            "ema":   {},   # window  → series
            "rsi":   {},   # length  → series
            "atr":   {},   # length  → series
            "macd":  {},   # length  → (macd, signal)
        }

        # External inputs
        self.account_balance = 1_000.0

    # ------------------------------------------------------------------ #
    #  UTILITY HELPERS
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_np(data: List[float]) -> np.ndarray:
        return np.asarray(data, dtype=np.float64)

    # Simple Moving Average ---------------------------------------------------
    def _sma(self, data: List[float], window: int) -> float:
        arr = self._to_np(data)
        if arr.size < window:
            return float(arr.mean())
        return float(arr[-window:].mean())

    # Exponential Moving Average ---------------------------------------------
    def _ema(self, data: List[float], window: int) -> np.ndarray:
        length = len(data)
        cache_key = window
        cached = self._cache["ema"].get(cache_key)
        if cached is not None and cached.size == length:
            return cached

        arr = self._to_np(data)
        alpha = 2.0 / (window + 1)
        ema = np.empty_like(arr)
        ema[0] = arr[0]
        for i in range(1, arr.size):
            ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]

        self._cache["ema"][cache_key] = ema
        return ema

    # Relative Strength Index -------------------------------------------------
    def _rsi(self, prices: List[float]) -> np.ndarray:
        length = len(prices)
        cache_key = length
        cached = self._cache["rsi"].get(cache_key)
        if cached is not None:
            return cached

        arr = self._to_np(prices)
        delta = np.diff(arr, prepend=arr[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        roll_up   = self._ewm_mean(gain, self.rsi_window)
        roll_down = self._ewm_mean(loss, self.rsi_window)
        rs        = np.divide(roll_up, roll_down, out=np.zeros_like(roll_up), where=roll_down != 0)
        rsi       = 100.0 - (100.0 / (1.0 + rs))

        self._cache["rsi"][cache_key] = rsi
        return rsi

    def _ewm_mean(self, arr: np.ndarray, window: int) -> np.ndarray:
        alpha = 1.0 / window
        ewm = np.empty_like(arr)
        ewm[0] = arr[0]
        for i in range(1, arr.size):
            ewm[i] = alpha * arr[i] + (1 - alpha) * ewm[i - 1]
        return ewm

    # MACD --------------------------------------------------------------------
    def _macd(self, prices: List[float]) -> tuple:
        length = len(prices)
        cached = self._cache["macd"].get(length)
        if cached is not None:
            return cached

        ema_fast = self._ema(prices, self.macd_fast)
        ema_slow = self._ema(prices, self.macd_slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line.tolist(), self.macd_signal)

        out = (macd_line, signal_line)
        self._cache["macd"][length] = out
        return out

    # Average True Range ------------------------------------------------------
    def _atr(self, highs: List[float], lows: List[float], closes: List[float]) -> np.ndarray:
            length = len(closes)
            cached = self._cache["atr"].get(length)
            if cached is not None:
                return cached

            h = self._to_np(highs)
            l = self._to_np(lows)
            c = self._to_np(closes)

            prev_close          = np.roll(c, 1)
            prev_close[0]       = c[0]
            true_range          = np.maximum.reduce([h - l,
                                                    np.abs(h - prev_close),
                                                    np.abs(l - prev_close)])

            atr = np.empty_like(true_range)
            # SMA of first *n* TRs
            atr[0] = true_range[: self.atr_window].mean()

            # Wilder’s smoothing
            for i in range(1, true_range.size):
                atr[i] = ((atr[i - 1] * (self.atr_window - 1)) + true_range[i]) / self.atr_window

            self._cache["atr"][length] = atr
            return atr

    # Volatility --------------------------------------------------------------
    def _calculate_volatility(self, prices: List[float]) -> float:
        arr = self._to_np(prices)
        if arr.size < 2:
            return 0.0
        returns = np.diff(np.log(arr))
        return float(returns[-self.volatility_window :].std(ddof=0) * np.sqrt(252))

    # Trend & Momentum --------------------------------------------------------
    def _trend_direction(self, ema_fast: np.ndarray, ema_slow: np.ndarray) -> str:
        if ema_fast[-1] > ema_slow[-1] and ema_fast[-2] <= ema_slow[-2]:
            return "bullish_crossover"
        if ema_fast[-1] < ema_slow[-1] and ema_fast[-2] >= ema_slow[-2]:
            return "bearish_crossover"
        return "bullish" if ema_fast[-1] > ema_slow[-1] else "bearish"

    def _momentum_strength(self, rsi: np.ndarray, macd: np.ndarray, signal: np.ndarray) -> Dict[str, str]:
        return {
            "rsi": "overbought" if rsi[-1] > 70 else "oversold" if rsi[-1] < 30 else "neutral",
            "macd_crossover": (
                "bullish"
                if macd[-1] > signal[-1] and macd[-2] <= signal[-2] else
                "bearish"
                if macd[-1] < signal[-1] and macd[-2] >= signal[-2] else
                "none"
            ),
        }

    def _volume_analysis(self, current_volume: float, volume_ma: float) -> Dict[str, object]:
        return {
            "volume_spike": current_volume > volume_ma * self.volume_spike_multiplier,
            "volume_trend": "rising" if current_volume > volume_ma else "falling",
        }

    # Dynamic SL/TP & Position Sizing ----------------------------------------
    def _dynamic_stop_loss_take_profit(self, price: float, atr: np.ndarray, volatility: float) -> tuple:
        sl = price - (atr[-1] * 1.5 * (1 + volatility))
        tp = price + (atr[-1] * 3.0 * (1 + volatility))
        return sl, tp

    def _calculate_position_size(self, sl: float, price: float, risk_percent: float) -> float:
        risk_amount = self.account_balance * risk_percent
        price_risk  = abs(price - sl)

        min_risk_distance = 0.20 * price_risk  # ensure at least 20 % of risk buffer
        if price_risk < min_risk_distance:
            return 0.0

        size = min(risk_amount / price_risk, self.account_balance * 0.1)
        return round(size, 2)

    # Recent Performance ------------------------------------------------------
    def _recent_performance(self) -> float:
        if len(self.trade_history) < 3:
            return 0.0
        wins = sum(1 for t in list(self.trade_history)[-3:] if t > 0)
        return (wins / 3.0) - 0.5

    # Signal Strength ---------------------------------------------------------
    def _signal_strength(self, trend: str, momentum: Dict, volume: Dict) -> Dict[str, object]:
        buy, sell = 0, 0
        if "bullish" in trend: buy += 1
        if momentum["macd_crossover"] == "bullish": buy += 1
        if momentum["rsi"] == "oversold": buy += 1
        if volume["volume_spike"] and volume["volume_trend"] == "rising": buy += 1

        if "bearish" in trend: sell += 1
        if momentum["macd_crossover"] == "bearish": sell += 1
        if momentum["rsi"] == "overbought": sell += 1
        if volume["volume_spike"] and volume["volume_trend"] == "falling": sell += 1

        if buy > sell:
            return {"direction": "BUY",  "strength": buy  / 4.0}
        if sell > buy:
            return {"direction": "SELL", "strength": sell / 4.0}
        return {"direction": "HOLD", "strength": 0.0}

    # Holding Time ------------------------------------------------------------
    def _holding_time(self, signal: Dict[str, object]) -> int:
        return int(max(1, min(5, 3 * (1 + signal["strength"]))))

    # Confidence --------------------------------------------------------------
        # ────────────────────────────────────────────────────────────
    # 3️⃣  confidence score – keep raw score then clamp
    # ────────────────────────────────────────────────────────────
    def _calculate_confidence(self, trend: str, momentum: Dict,
                               volume: Dict, perf: float) -> float:

        raw = 0.0
        # trend weight
        raw += 0.25 if "crossover" in trend else 0.10 if trend in ("bullish", "bearish") else 0.0
        # momentum
        raw += 0.20 if momentum["rsi"] in ("overbought", "oversold") else 0.0
        raw += 0.10 if momentum["macd_crossover"] != "none" else 0.0
        # volume
        raw += 0.15 if volume["volume_spike"] else 0.0
        raw += 0.05 if volume["volume_trend"] == "rising" else 0.0

        raw *= (1 + perf)           # recent P/L adjustment
        conf = float(np.clip(raw, self.min_confidence, self.max_confidence))
        return conf


    # ------------------------------------------------------------------ #
    #  PUBLIC ENTRY
    # ------------------------------------------------------------------ #
    def generate_candle_signal(self, candle_data: List[Dict[str, float]]) -> Dict[str, object]:
        if len(candle_data) < max(self.macd_slow, self.atr_window, self.rsi_window):
            return self._hold("Insufficient data")

        # Unpack arrays
        closes  = [c["close"]  for c in candle_data]
        highs   = [c["high"]   for c in candle_data]
        lows    = [c["low"]    for c in candle_data]
        volumes = [c["volume"] for c in candle_data]

        ema_fast  = self._ema(closes, 9)
        ema_slow  = self._ema(closes, 21)
        rsi_arr   = self._rsi(closes)
        macd_arr, signal_arr = self._macd(closes)
        atr_arr   = self._atr(highs, lows, closes)
        vol_ma    = self._sma(volumes, 20)
        volatility = self._calculate_volatility(closes)

        sl, tp = self._dynamic_stop_loss_take_profit(closes[-1], atr_arr, volatility)
        trend   = self._trend_direction(ema_fast, ema_slow)
        mom     = self._momentum_strength(rsi_arr, macd_arr, signal_arr)
        vol_an  = self._volume_analysis(volumes[-1], vol_ma)
        perf    = self._recent_performance()
        conf    = self._calculate_confidence(trend, mom, vol_an, perf)
        signal  = self._signal_strength(trend, mom, vol_an)
        size    = self._calculate_position_size(sl, closes[-1], self.risk_per_trade)

        return {
            "signal":           signal["direction"],
            "stop_loss":        sl,
            "take_profit":      tp,
            "confidence":       conf,
            "position_size":    size,
            "max_holding_time": self._holding_time(signal),
            "trailing_stop":    atr_arr[-1] * 2.0,
            "indicators": {
                "ema_fast":  float(ema_fast[-1]),
                "ema_slow":  float(ema_slow[-1]),
                "rsi":       float(rsi_arr[-1]),
                "macd":      float(macd_arr[-1]),
                "signal_ln": float(signal_arr[-1]),
                "atr":       float(atr_arr[-1]),
            },
        }

    # ------------------------------------------------------------------ #
    #  INTERNAL: HOLD PAYLOAD
    # ------------------------------------------------------------------ #
    def _hold(self, reason: str) -> Dict[str, object]:
        if self.logger:
            self.logger.debug(f"HOLD – {reason}")
        return {
            "signal": "HOLD",
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "confidence": 0.0,
            "position_size": 0.0,
            "max_holding_time": 0,
            "trailing_stop": 0.0,
            "indicators": {},
        }
