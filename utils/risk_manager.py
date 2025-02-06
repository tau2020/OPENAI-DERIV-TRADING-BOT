# risk_manager.py
import logging
from datetime import datetime
from typing import Optional


class RiskManager:
    """
    RiskManager implements risk control checks and position sizing logic.
    It maintains risk state (daily loss/profit, consecutive losses) and
    ensures that trading does not exceed preset risk limits.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        max_daily_drawdown: float = 0.03,
        max_consecutive_losses: int = 3,
        daily_loss_limit_percent: float = 0.02,
        daily_profit_target_percent: float = 0.03
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.max_daily_drawdown = max_daily_drawdown
        self.max_consecutive_losses = max_consecutive_losses
        self.daily_loss_limit_percent = daily_loss_limit_percent
        self.daily_profit_target_percent = daily_profit_target_percent

        # Risk tracking
        self.daily_start_balance = 0.0
        self.current_balance = 0.0
        self.consecutive_losses = 0
        self.last_update = datetime.now()

    def initialize_account(self, starting_balance: float) -> None:
        """Call at the beginning of the trading day."""
        self.daily_start_balance = starting_balance
        self.current_balance = starting_balance
        self.consecutive_losses = 0
        self.logger.info(f"Risk Manager initialized: Start balance = {starting_balance}")

    

    def record_trade(self, profit: float) -> None:
        """Record a trade result for risk tracking."""
        if profit < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def risk_limits_reached(self) -> bool:
        """Check if any risk limit is breached."""
        # Check drawdown
        peak_balance = max(self.daily_start_balance, self.current_balance)
        if peak_balance <= 0:
            return True

        drawdown = (peak_balance - self.current_balance) / peak_balance
        if drawdown >= self.max_daily_drawdown:
            self.logger.warning(f"Drawdown limit reached: {drawdown:.2%}")
            return True

        # Check daily loss limit (or profit target reached)
        if (self.daily_start_balance - self.current_balance) / self.daily_start_balance >= self.daily_loss_limit_percent:
            self.logger.warning("Daily loss limit reached")
            return True

        if (self.current_balance - self.daily_start_balance) / self.daily_start_balance >= self.daily_profit_target_percent:
            self.logger.info("Daily profit target reached")
            return True

        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.logger.warning(f"Consecutive losses limit reached: {self.consecutive_losses}")
            return True

        return False

    def get_risk_info(self) -> dict:
        return {
            "daily_start_balance": self.daily_start_balance,
            "current_balance": self.current_balance,
            "consecutive_losses": self.consecutive_losses,
            "max_daily_drawdown": self.max_daily_drawdown
        }
        
        
    def update_balance(self, new_balance: float):
        self.current_balance = new_balance
        self.logger.info("Updated account balance: %.2f", new_balance)

    def calculate_position_size(self, current_price: float, stop_loss: float) -> float:
        """
        Calculates the number of shares (or contracts) to buy/sell based on the risk per trade.
        """
        risk_per_share = abs(current_price - stop_loss)
        if risk_per_share == 0:
            self.logger.warning("Risk per share is zero; cannot calculate position size.")
            return 0.0
        risk_amount = self.current_balance * self.risk_per_trade
        position_size = risk_amount / risk_per_share
        # Cap the position size if necessary.
        position_size = min(position_size, self.max_position_size)
        self.logger.info("Calculated position size: %.2f", position_size)
        return position_size    
