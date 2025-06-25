"""
FastAPI REST layer with enhanced logging for frontend monitoring.

Fix applied:
    - Added a log capturing system using a deque.
    - Exposed a new endpoint `/strategy/logs` for the frontend to fetch logs.
    - Enriched the startup process with detailed logging for better user feedback.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import deque
from typing import List

from fastapi import BackgroundTasks, FastAPI, HTTPException
from starlette.datastructures import State
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from deriv_api import DerivAPI
from config import config
from strategy.trading_strategy import TradingStrategy

# ────────────────────────── Log Capture Setup ──────────────────────────── #

# Create a deque to store recent log messages in memory
log_capture = deque(maxlen=100)

# Custom handler to push log records into our deque
class DequeHandler(logging.Handler):
    def __init__(self, deque_instance: deque):
        super().__init__()
        self.deque_instance = deque_instance

    def emit(self, record):
        self.deque_instance.append(self.format(record))

# Configure logging
log_formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("trading-api")
logger.setLevel(logging.INFO)

# Add the custom deque handler to the logger
deque_handler = DequeHandler(log_capture)
deque_handler.setFormatter(log_formatter)
logger.addHandler(deque_handler)

# Optional: Add a stream handler to also see logs in the console
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)


# ────────────────────────── lifespan handler ──────────────────────────── #

async def lifespan(app: FastAPI):
    """Create DerivAPI + TradingStrategy singletons and tear them down cleanly."""
    app.state: State  # type‑hint for linters

    api_token = getattr(config, "DERIV_API_TOKEN", os.getenv("DERIV_API_TOKEN"))
    if not api_token:
        raise RuntimeError("DERIV_API_TOKEN missing – set in config or env var")

    app_id = 68054
    api_client = DerivAPI(app_id=app_id)

    # Note: Authorization and other setup steps are now moved into the background task
    # to provide real-time feedback to the user.
    app.state.deriv_api = api_client
    app.state.strategy = TradingStrategy(
        logger=logger,
        deriv_api_client=app.state.deriv_api,
        loginid=getattr(config, "LOGIN_ID", "Unknown"),
    )
    app.state.streaming_task: asyncio.Task | None = None

    logger.info("Application ready. Waiting for user to start strategy.")
    yield
    # Shutdown
    if app.state.streaming_task and not app.state.streaming_task.done():
        app.state.streaming_task.cancel()

    if app.state.deriv_api.has_connection:
        await app.state.deriv_api.disconnect()
    logger.info("Shutdown complete – API disconnected")


app = FastAPI(
    title="Deriv Trading Bot API",
    version="0.2.0",
    lifespan=lifespan,
)

origins = ["http://localhost", "http://localhost:5173", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────── pydantic schemas ──────────────────────────── #

class StartRequest(BaseModel):
    symbols: List[str] | None = None

# ... (other schemas remain the same) ...

# ────────────────────────── helper functions ──────────────────────────── #

def _is_running() -> bool:
    task: asyncio.Task | None = app.state.streaming_task
    return task is not None and not task.done()

async def _run_strategy() -> None:
    """The main background task with detailed logging for the frontend."""
    strategy: TradingStrategy = app.state.strategy
    api_client: DerivAPI = app.state.deriv_api
    api_token = getattr(config, "DERIV_API_TOKEN", os.getenv("DERIV_API_TOKEN"))

    try:
        # Step 1: Authorize
        logger.info("Authorizing with Deriv API...")
        authorize_response = await api_client.authorize(api_token)
        if 'error' in authorize_response:
            logger.error(f"Authorization failed: {authorize_response['error']['message']}")
            return
        logger.info("Authorization successful.")

        # Step 2: Set Active Symbols
        if not strategy.active_symbols:
            logger.info(f"No symbols provided, using default: {config.SYMBOL}")
            strategy.set_active_symbols([config.SYMBOL])
        else:
            logger.info(f"Using provided symbols: {', '.join(strategy.active_symbols)}")

        # Step 3: Fetch initial account info
        await strategy.log_account_info()

        # Step 4: Subscribe to ticks
        logger.info(f"Subscribing to ticks for {len(strategy.active_symbols)} symbol(s)...")
        await asyncio.gather(*(strategy.subscribe_live_ticks(s) for s in strategy.active_symbols))
        logger.info("All tick subscriptions are live. Strategy is now running.")

    except Exception as e:
        logger.error(f"An error occurred during strategy startup: {e}")
        # Ensure the task is considered "done" on failure
        raise

# ───────────────────────────── endpoints ──────────────────────────────── #

@app.get("/strategy/logs")
async def get_logs():
    """Endpoint to fetch captured log messages."""
    return {"logs": list(log_capture)}

@app.post("/strategy/start")
async def start_strategy(body: StartRequest, bg: BackgroundTasks):
    if _is_running():
        raise HTTPException(status_code=400, detail="Strategy already running")

    # Clear previous logs for the new session
    log_capture.clear()

    strat: TradingStrategy = app.state.strategy
    if body.symbols:
        strat.set_active_symbols(body.symbols)

    # Start the _run_strategy task in the background
    task = asyncio.create_task(_run_strategy(), name="strategy-runner")
    app.state.streaming_task = task
    bg.add_task(lambda: task)

    return {"status": "starting", "symbols": strat.active_symbols}

# ... (the rest of your endpoints: /stop, /status, /trades, etc. remain the same) ...
@app.post("/strategy/stop")
async def stop_strategy():
    if not _is_running():
        raise HTTPException(status_code=400, detail="Strategy not running")

    app.state.streaming_task.cancel()
    logger.info("Strategy stop request received.")
    await asyncio.sleep(0)
    return {"status": "stopped"}

@app.get("/strategy/status")
async def strategy_status():
    strat: TradingStrategy = app.state.strategy
    return {
        "running": _is_running(),
        "symbols": strat.active_symbols,
        "balance": strat.latest_balance,
        "wins": strat.total_wins,
        "losses": strat.total_losses,
        "net_pl": strat.cumulative_profit - strat.cumulative_loss,
        "drawdown": getattr(strat, "daily_pl", 0.0),
    }

@app.get("/strategy/trades")
async def trades():
    return app.state.strategy.trades_data


# This endpoint already exists and will now use our more efficient method.
@app.get("/strategy/open-contracts")
async def open_contracts():
    """Returns a list of all currently open contracts."""
    strat: TradingStrategy = app.state.strategy
    if not _is_running():
        # It's useful to see open contracts even if the bot is stopped.
        # You might have open positions from a previous session.
        pass  # Allow the call to proceed even if the bot is stopped.

    return await strat.get_open_contracts_info()


# This endpoint also already exists, ready to be used.
@app.post("/strategy/sell/{contract_id}")
async def sell_contract(contract_id: str):
    """Triggers a sell order for a specific contract_id."""
    strat: TradingStrategy = app.state.strategy
    # Ensure contract_id is an integer, as required by the sell() call
    try:
        contract_id_int = int(contract_id)
        return await strat.api_sell_contract(contract_id_int)
    except ValueError:
        raise HTTPException(status_code=400, detail="Contract ID must be an integer.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/strategy/metrics")
async def metrics():
    strat: TradingStrategy = app.state.strategy
    return {
        "daily_start_balance": strat.daily_start_balance,
        "latest_balance": strat.latest_balance,
        "daily_pl": getattr(strat, "daily_pl", 0.0),
        "cumulative_profit": strat.cumulative_profit,
        "cumulative_loss": strat.cumulative_loss,
        "consecutive_losses": strat.consecutive_losses,
        "volatility": strat.volatility,
        "atr": strat.average_true_range,
    }

@app.post("/strategy/symbols")
async def set_symbols(body: StartRequest): # Using StartRequest for symbols
    strat: TradingStrategy = app.state.strategy
    strat.set_active_symbols(body.symbols)
    return {"symbols": strat.active_symbols}

@app.post("/strategy/hedge-threshold")
async def hedge_threshold(body: StartRequest): # Assuming it has a threshold field, should create a proper schema
    # This needs a proper Pydantic schema like HedgeThresholdRequest
    pass
