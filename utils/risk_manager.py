# utils/risk_manager.py
import logging
from datetime import datetime

class RiskManager:
    """
    Implements risk control checks and position sizing logic.
    Maintains risk state (daily P/L, consecutive losses) and ensures trading
    does not exceed preset risk limits.
    """
    def __init__(
        self,
        logger: logging.Logger = None,
        max_daily_drawdown: float = 50.63,
        max_consecutive_losses: int = 50,
        daily_loss_limit_percent: float = 50.72,
        daily_profit_target_percent: float = 30.03,
        risk_per_trade: float = 0.01,   # 1% of balance per trade
        max_position_size: float = 1000 # maximum allowed position size
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.max_daily_drawdown = max_daily_drawdown
        self.max_consecutive_losses = max_consecutive_losses
        self.daily_loss_limit_percent = daily_loss_limit_percent
        self.daily_profit_target_percent = daily_profit_target_percent
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size

        self.daily_start_balance = 0.0
        self.current_balance = 0.0
        self.consecutive_losses = 0
        self.last_update = datetime.now()

    def initialize_account(self, starting_balance: float) -> None:
        """Initializes the risk manager at the start of the trading day."""
        self.daily_start_balance = starting_balance
        self.current_balance = starting_balance
        self.consecutive_losses = 0
        self.logger.info(f"Risk Manager initialized: Start balance = {starting_balance}")

    def record_trade(self, profit: float) -> None:
        """Records a trade result for risk tracking."""
        self.current_balance += profit
        if profit < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        self.logger.info(f"Trade recorded: profit={profit:.2f}, current_balance={self.current_balance:.2f}")

    def risk_limits_reached(self) -> bool:
        """Checks if any risk limit is breached."""
        peak_balance = max(self.daily_start_balance, self.current_balance)
        if peak_balance <= 0:
            return True

        drawdown = (peak_balance - self.current_balance) / peak_balance
        if drawdown >= self.max_daily_drawdown:
            self.logger.warning(f"Drawdown limit reached: {drawdown:.2%}")
            return True

        if (self.daily_start_balance - self.current_balance) / self.daily_start_balance >= self.daily_loss_limit_percent:
            self.logger.warning("Daily loss limit reached")
            return True

        if (self.current_balance - self.daily_start_balance) / self.daily_start_balance >= self.daily_profit_target_percent:
            self.logger.info("Daily profit target reached")
            return True

        if self.consecutive_losses >= self.max_consecutive_losses:
            self.logger.warning(f"Consecutive losses limit reached: {self.consecutive_losses}")
            return True

        return False

    def update_balance(self, new_balance: float):
        self.current_balance = new_balance
        self.logger.info("Updated account balance: %.2f", new_balance)

    def calculate_position_size(self, current_price: float, stop_loss: float) -> float:
        """
        Calculates the position size (number of contracts) based on the risk per trade.
        """
        risk_per_unit = abs(current_price - stop_loss)
        if risk_per_unit == 0:
            self.logger.warning("Risk per unit is zero; cannot calculate position size.")
            return 0.0
        risk_amount = self.current_balance * self.risk_per_trade
        position_size = risk_amount / risk_per_unit
        position_size = min(position_size, self.max_position_size)
        self.logger.info("Calculated position size: %.2f", position_size)
        return position_size

    def get_risk_info(self) -> dict:
        return {
            "daily_start_balance": self.daily_start_balance,
            "current_balance": self.current_balance,
            "consecutive_losses": self.consecutive_losses,
            "max_daily_drawdown": self.max_daily_drawdown
        }
