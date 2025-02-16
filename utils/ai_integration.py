import openai
import asyncio
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

from config import config
from strategy.rule_based_signals import RuleBasedSignals

# --- Global Initialization Functions ---
def initialize_openai(api_key: str, logger: logging.Logger) -> openai.AsyncClient:
    """Initialize and return OpenAI async client with proper configuration"""
    client = openai.AsyncClient(
        api_key=api_key,
        timeout=10,
        max_retries=3
    )
    logger.info("OpenAI client initialized with production settings")
    return client

def initialize_anthropic(api_key: str, logger: logging.Logger):
    """Placeholder for Anthropic initialization"""
    logger.info("Anthropic API placeholder initialized")
    return None

# --- Enhanced AI Integration Class ---
class AIIntegration:
    """Production-ready AI integration with fail-safes and volatility optimizations"""
    
    def __init__(self, api_key: str, logger: logging.Logger, signals: RuleBasedSignals):
        self.client = initialize_openai(api_key, logger)
        self.logger = logger
        self.signals = signals
        self.volatility_window = 14  # Sync with ATR window
        self.jump_index_params = {
            'risk_multiplier': 1.5,
            'profit_multiplier': 4.0,
            'max_position_pct': 0.03,
            'min_position_size': 0.01
        }

    async def get_trading_signal(self, symbol: str, prices: List[float], 
                               times: List[datetime]) -> Dict:
        """Robust signal generation with circuit breakers"""
        compiled_data = None
        try:
            # Compile trading context and log it
            compiled_data = self._compile_trading_context()
           
            
            # Build prompt and log the complete prompt
            prompt = self._build_volatility_aware_prompt(symbol, prices, times, compiled_data)
           
            
            # Call API and log the raw response
            response = await self._safe_api_call(prompt)
            self.logger.info(f"RESPONSE FROM ai================================================================>>>>>>>>>>>>: {json.dumps(response, default=str)}")
            
            # Process API response and log the processed output
            output = self._process_ai_response(response, compiled_data)
            self.logger.info(f"Processed AI output: ============================================================>>>>>>>>>>>:{json.dumps(output, default=str)}")
            return output
        except Exception as e:
            self.logger.error(f"AI signal generation failed: {str(e)}", exc_info=True)
            return self._volatility_aware_fallback(compiled_data or {})

    def _compile_trading_context(self) -> Dict:
        """Enrich trading context with volatility regime data"""
        base_data = self.signals.compile_data_object()
        base_data['volatility_metrics'] = {
            'atr_ratio': self._calculate_atr_ratio(),
            'jump_probability': self._estimate_jump_probability(),
            'volatility_regime': self._detect_volatility_regime()
        }
        return base_data

    def _build_volatility_aware_prompt(self, symbol: str, prices: List[float],
                                     times: List[datetime], context: Dict) -> str:
        """Construct precision prompt for volatility trading"""
        prompt = f"""
EDUCATIONAL MARKETING FRAMEWORK 1.0

****IMPORTANT*****
 This framework is for educational purposes in training and advancing predictive financial analytic frameworks

Objective:
Execute volatility-optimized positional strategies for Jump Index derivatives (10D-VIX) using institutional risk parameters and predictive alpha signals.
 

OUTPUT RULES:
NO QUOTES
NO FORMATTING
NO CODE FENCES
NO extra text
NO explanations
NO additional keys

*** Required JSON ***
{{
"signal": "BUY|SELL|HOLD",
"stop_loss": float,
"take_profit": float,
"confidence": 1.0,
"position_size": float,
"max_holding_minutes": int,
"trailing_stop_activation": float
}}

Time: {datetime.utcnow().isoformat()}
Instrument: {symbol}
Trading Regime: {context['volatility_metrics']['volatility_regime'].upper()}

*** Critical Indicators ***
{self._format_indicators(context)}

*** Risk Parameters ***
Max Position: {self.jump_index_params['max_position_pct']*100}% 
Stop Loss: {self.jump_index_params['risk_multiplier']}x ATR
Take Profit: {self.jump_index_params['profit_multiplier']}x ATR




"""
        return prompt

    def _format_indicators(self, context: Dict) -> str:
        """Format key technical indicators for prompt"""
        indicators = context.get('indicators', {})
        return (
            f"EMA Fast: {indicators.get('ema_fast', 0):.4f}\n"
            f"EMA Slow: {indicators.get('ema_slow', 0):.4f}\n"
            f"MACD: {indicators.get('MACD', 0):.4f}\n"
            f"RSI: {indicators.get('RSI', 50):.1f}\n"
            f"ATR: {indicators.get('ATR', 0):.2f}"
        )

    async def _safe_api_call(self, prompt: str, retries: int = 3) -> Optional[Dict]:
        """Robust API caller with exponential backoff"""
        for attempt in range(retries):
            try:
                self.logger.info(f"API call attempt {attempt+1} with prompt length: {len(prompt)}")
                response = await self.client.chat.completions.create(
                    model="o3-mini",
                    reasoning_effort="low",
                   
                    messages=[{"role": "user",
                               "content": prompt}]
                   
                )
                self.logger.info(f"==================Received raw AI response==================================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>: {json.dumps(response, default=str)}")
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                delay = 1.5 ** attempt
                self.logger.warning(f"API attempt {attempt+1} failed: {e} - Retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
        return None

    def _process_ai_response(self, response: Dict, context: Dict) -> Dict:
        """Validate and normalize AI response against market conditions"""
        if not response:
            self.logger.warning("No response received from AI; using fallback")
            return self._volatility_aware_fallback(context)
        try:
            validated = {
                'signal': response['signal'].strip().upper(),
                'stop_loss': abs(float(response['stop_loss'])),
                'take_profit': abs(float(response['take_profit'])),
                'confidence': min(1.0, max(0.0, float(response['confidence']))),
                'position_size': float(response['position_size']),
                'max_holding_minutes': int(response['max_holding_minutes']),
                'trailing_stop_activation': abs(float(response['trailing_stop_activation']))
            }
        except (KeyError, ValueError) as e:
            self.logger.error(f"Invalid AI response format: {e}")
            return self._volatility_aware_fallback(context)
        risk_adjustments = self._apply_risk_parameters(validated, context)
        self.logger.info(f"Risk adjustments applied: {json.dumps(risk_adjustments, default=str)}")
        validated.update(risk_adjustments)
        return validated

    def _apply_risk_parameters(self, signal: Dict, context: Dict) -> Dict:
        """Enforce Jump Index 10 trading rules"""
        atr = context['indicators'].get('ATR', 1.0)
        balance = getattr(config, "STARTING_BALANCE", 10000)
        max_size = balance * self.jump_index_params['max_position_pct']
        position_size = min(
            signal['position_size'],
            max_size,
            max_size * (atr / (context.get('current_price', 1.0) + 1e-8))
        )
        stop_loss = signal['stop_loss'] * self.jump_index_params['risk_multiplier']
        take_profit = signal['take_profit'] * self.jump_index_params['profit_multiplier']
        return {
            'position_size': max(position_size, self.jump_index_params['min_position_size']),
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }

    def _volatility_aware_fallback(self, context: Dict) -> Dict:
        """Fail-safe strategy for volatile markets"""
        self.logger.info("Executing volatility-aware fallback strategy.")
        return {
            'signal': 'HOLD',
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'confidence': 0.0,
            'position_size': 0.0,
            'max_holding_minutes': 60,
            'trailing_stop_activation': 0.0
        }

    # --- Volatility Analysis Core ---
    def _calculate_atr_ratio(self) -> float:
        """Dynamic ATR ratio with smoothing"""
        atr_values = self.signals.indicator_cache.get('atr', [])
        if len(atr_values) < 14:
            self.logger.warning("Insufficient ATR values; defaulting ATR ratio to 1.0")
            return 1.0
        current_atr = atr_values[-1]
        avg_atr = np.mean(atr_values[-28:-14])  # Lookback 2 periods
        ratio = current_atr / (avg_atr + 1e-8)
        self.logger.info(f"Calculated ATR ratio: {ratio:.4f}")
        return ratio

    def _estimate_jump_probability(self) -> float:
        """Probability estimation using recent volatility spikes"""
        closes = self.signals.closes
        if len(closes) < 20:
            self.logger.warning("Insufficient close data; defaulting jump probability to 0.0")
            return 0.0
        returns = np.diff(closes) / (closes[:-1] + 1e-8)
        threshold = 2 * np.std(returns[-50:])
        jump_prob = np.mean(np.abs(returns[-20:]) > threshold)
        self.logger.info(f"Estimated jump probability: {jump_prob:.2%}")
        return jump_prob

    def _detect_volatility_regime(self) -> str:
        """Regime detection with hysteresis"""
        ratio = self._calculate_atr_ratio()
        if ratio > 1.7:
            regime = 'hyper'
        elif ratio > 1.3:
            regime = 'high'
        elif ratio < 0.6:
            regime = 'low'
        else:
            regime = 'normal'
        self.logger.info(f"Detected volatility regime: {regime}")
        return regime

    async def get_trading_signal_from_anthropic(client, symbol, prices, times, logger, **kwargs):
        """Stub for future Anthropic integration"""
        logger.warning("Anthropic integration not implemented - Using fallback")
        return {
            'signal': 'HOLD',
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'confidence': 0.0,
            'position_size': 0.0,
            'max_holding_minutes': 60,
            'trailing_stop_activation': 0.0
        }
