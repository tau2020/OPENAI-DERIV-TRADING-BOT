import os
import json
import logging

import pandas as pd
import anthropic
import asyncio

class ClaudeTradingClient:
    def __init__(self, logger: logging.Logger, api_key: str = None, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initializes the ClaudeTradingClient with a logger and optional API key and model.
        If no key is provided, it uses ANTHROPIC_API_KEY from the environment.
        """
        self.logger = logger
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key must be provided.")

        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.logger.info("Anthropic Claude API client initialized.")

    async def get_trading_signal(
        self,
        symbol: str,
        prices: list,
        times: list,
        recent_volume=None,
        recent_volatility=None,
        sentiment_score=None,
        recent_trades_win_loss_ratio=None,
        account_balance=None,
        news_headlines=None,
        polygon_data=None,
        temperature=0,
        max_tokens=150
    ) -> dict:
        """
        Requests a trading signal from Claude given market and sentiment data.
        Returns a dict with keys:
        signal, stop_loss, take_profit, confidence, position_size, max_holding_time, trailing_stop
        or a HOLD fallback if an error occurs.
        """

        # Provide defaults if fields are None
        recent_volume = recent_volume if recent_volume is not None else []
        recent_volatility = recent_volatility if recent_volatility is not None else 0.0
        sentiment_score = sentiment_score if sentiment_score is not None else 0.0
        recent_trades_win_loss_ratio = recent_trades_win_loss_ratio if recent_trades_win_loss_ratio is not None else 1.0
        account_balance = account_balance if account_balance is not None else 1000.0
        news_headlines = news_headlines or []
        polygon_data = polygon_data or "No additional market data."

        # Truncate data points
        max_data_points = 150
        if len(prices) > max_data_points:
            self.logger.info(f"Truncating input data from {len(prices)} to {max_data_points} points")
            prices = prices[-max_data_points:]
            times = times[-max_data_points:]

        price_data_str = "\n".join(
            [f"{pd.Timestamp(t).strftime('%Y-%m-%d %H:%M:%S')}, {p}" for t, p in zip(times, prices)]
        )
        news_str = "\n".join(news_headlines[:3])

        system_message = """\You are a professional quantitative trading analyst focused on high-probability setups and strict risk management.

                                    TRADING FRAMEWORK:
                                    Version: 3.0
                                    Type: Advanced Multi-Factor Technical & Risk-Adjusted Analysis

                                    TECHNICAL ANALYSIS CRITERIA:
                                    1. Trend Analysis:
                                    - Primary: EMA crossovers (20/50/200)
                                    - Secondary: VWAP deviation & Bollinger Bands (2 std)
                                    - Volume confirmation required for all signals
                                    - Minimum 2:1 reward/risk ratio for all trades

                                    2. Entry Conditions (ALL required for BUY/SELL):
                                    - Clear trend direction (20 EMA vs 50 EMA slope)
                                    - Volume > 1.5x 20-period average
                                    - Price respecting key support/resistance
                                    - RSI(14) not in extreme zones (>70 or <30)
                                    - Clear price action confirmation (no shooting stars/hammers against trend)

                                    3. Risk Management Rules:
                                    - Position size: 2-5% of account per trade
                                    - Stop loss: Based on recent ATR(14) * 1.5
                                    - Take profit: Minimum 2x stop loss distance
                                    - Trailing stop: Activated at 1x risk reached
                                    - Scale out: 50% at 1.5x risk, trail remainder

                                    4. Volatility Requirements:
                                    - ATR within 20-period average ±30%
                                    - No trades during extreme volatility
                                    - Adjust position size inversely with volatility

                                    5. Confidence Scoring:
                                    Base: 0.5
                                    Add:
                                    +0.1: Strong trend alignment
                                    +0.1: Volume confirmation
                                    +0.1: Key level test
                                    +0.1: Positive sentiment
                                    +0.1: Low volatility
                                    +0.1: Strong financials/news
                                    +0.1: Historical win rate >60%

                                    Subtract:
                                    -0.1: Conflicting signals
                                    -0.1: High volatility
                                    -0.1: Poor volume
                                    -0.1: Negative sentiment
                                    -0.1: Against major trend
                                    -0.2: Near major event/news

                                    EXECUTION RULES:
                                    1. Signal Requirements:
                                    - "BUY": Uptrend, support test, volume surge
                                    - "SELL": Downtrend, resistance test, volume surge
                                    - "HOLD": Any uncertainty or criteria not met

                                    2. Position Sizing:
                                    - Base size: Account * 0.02 * confidence
                                    - Adjust for volatility: size * (1/ATR_ratio)
                                    - Minimum size: 1 unit
                                    - Maximum size: 5% of account

                                    3. Trade Management:
                                    - Initial stop: Key level or ATR-based
                                    - Take profit: Multiple price targets
                                    - Time stop: Based on volatility cycle
                                    - Trailing stop: Dynamic ATR-based

                                    OUTPUT REQUIREMENTS:
                                    1. Return ONLY a JSON object with exact format:
                                    {
                                        "signal": "BUY"|"SELL"|"HOLD",
                                        "stop_loss": float,
                                        "take_profit": float,
                                        "confidence": float,
                                        "position_size": float,
                                        "max_holding_time": int,
                                        "trailing_stop": float
                                    }

                                    2. Strict Output Rules:
                                    - No markdown/formatting
                                    - No explanations
                                    - Only pure JSON
                                    - No null values
                                    - Position size ≥ 1.0
                                    - Confidence between 0.0-1.0
                                    - Holding time ≥ 60 (minutes)
                                    - All stops/targets as floating point
                                    - Only uppercase signals

                                    3. Default Values:
                                    - Missing numeric: 0.0 (float), 60 (int)
                                    - Uncertain signal: "HOLD"
                                    - Low position size: 1.0 minimum
                                    - Invalid confidence: 0.0
                                    """
        user_message = f"""
                                    DATA INPUTS:
                                    - Symbol: {symbol}
                                    - Historical Price Data: {price_data_str}
                                    - Recent Volume: {recent_volume}
                                    - Volatility: {recent_volatility}
                                    - Sentiment: {sentiment_score}
                                    - W/L Ratio: {recent_trades_win_loss_ratio}, Balance: {account_balance}
                                    - Polygon: {polygon_data}, News: {news_str}

                                    Based on the above data, determine the best trading signal (BUY, SELL, or HOLD) along with recommended parameters:
                                    - Choose signal based on short-term technical and sentiment factors.
                                    - Provide stop_loss and take_profit as price levels relative to current price from the data.
                                    - confidence should be between 0.0 and 1.0.
                                    - Ensure position_size ≥ 0.35.
                                    - max_holding_time in seconds or minutes as an integer (default 60 if unsure).
                                    - trailing_stop as a float representing price distance.

                                    Return only the JSON object as per the system instructions.
                                    """

        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_message,
                messages=[
                    {
                        "role": "user",
                        "content": user_message.strip()
                    }
                ]
            )

            raw_content = response.content
            self.logger.info(f"Claude raw response: {raw_content}")

            # Handle if raw_content is a list of text blocks
            if isinstance(raw_content, list):
                # Join all text fields
                content_str = "".join(block.text for block in raw_content if hasattr(block, 'text'))
            else:
                content_str = str(raw_content)

            content_str = content_str.strip()

            # Remove code fences if present
            if content_str.startswith("```"):
                # Strip backticks and remove ```json tags if any
                content_str = content_str.strip("`")
                content_str = content_str.replace("```json", "").replace("```", "").strip()

            if not content_str:
                self.logger.error("Claude returned empty response. Defaulting to HOLD.")
                return self._fallback_hold_signal()

            try:
                result = json.loads(content_str)
                required_keys = [
                    "signal", "stop_loss", "take_profit", "confidence",
                    "position_size", "max_holding_time", "trailing_stop"
                ]
                for k in required_keys:
                    if k not in result:
                        self.logger.error(f"Missing key '{k}' in Claude response. Defaulting to HOLD.")
                        return self._fallback_hold_signal()

                signal = (result.get("signal") or "HOLD").upper()
                if signal not in ["BUY", "SELL", "HOLD"]:
                    self.logger.error(f"Invalid signal: {signal}. Defaulting to HOLD.")
                    signal = "HOLD"

                # Safely parse numeric fields, default if None
                stop_loss = float(result.get("stop_loss") or 0.0)
                take_profit = float(result.get("take_profit") or 0.0)
                confidence = float(result.get("confidence") or 0.0)
                position_size = float(result.get("position_size") or 0.0)
                max_holding_time = float(result.get("max_holding_time") or 60.0)
                trailing_stop = float(result.get("trailing_stop") or 0.0)

                # Ensure minimum stake >= 0.35
                min_stake = 0.35
                if position_size < min_stake:
                    # If you have config.AMOUNT available, you can choose max(min_stake, config.AMOUNT)
                    # Otherwise, just use min_stake
                    position_size = max(position_size, min_stake)

                final_result = {
                    "signal": signal,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "confidence": confidence,
                    "position_size": position_size,
                    "max_holding_time": max_holding_time,
                    "trailing_stop": trailing_stop
                }

                self.logger.info(f"Claude recommendation: {final_result}")
                return final_result

            except json.JSONDecodeError:
                self.logger.error("Unable to parse JSON from Claude response. Defaulting to HOLD.")
                return self._fallback_hold_signal()

        except Exception as e:
            self.logger.error(f"Error calling Claude API: {e}")
            return self._fallback_hold_signal()

    def _fallback_hold_signal(self) -> dict:
        """Return a default HOLD signal if an error occurs."""
        return {
            "signal": "HOLD",
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "confidence": 0.0,
            "position_size": 0.0,
            "max_holding_time": 60,
            "trailing_stop": 0.0
        }
