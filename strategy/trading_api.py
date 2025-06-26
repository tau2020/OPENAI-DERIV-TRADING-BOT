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

from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket
from starlette.datastructures import State
from starlette.websockets import WebSocketDisconnect
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from deriv_api import DerivAPI
from config import config
from strategy.trading_strategy import TradingStrategy


# 1. WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast_json(self, json_data: dict | list):
        for connection in self.active_connections:
            await connection.send_json(json_data)


portfolio_manager = ConnectionManager()

# 2. Log Capture Setup
log_capture = deque(maxlen=100)


class DequeHandler(logging.Handler):
    def __init__(self, deque_instance: deque):
        super().__init__()
        self.deque_instance = deque_instance

    def emit(self, record):
        self.deque_instance.append(self.format(record))


log_formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("trading-api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    deque_handler = DequeHandler(log_capture)
    deque_handler.setFormatter(log_formatter)
    logger.addHandler(deque_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)


# ────────────────────────── Helper Functions ──────────────────────────── #

async def broadcast_portfolio_updates():
    """Fetches fresh contract data and pushes it to all WebSocket clients."""
    # This check is important to prevent errors if the strategy hasn't been initialized yet.
    if hasattr(app.state, "strategy") and app.state.strategy:
        contracts = await app.state.strategy.get_open_contracts_info()
        await portfolio_manager.broadcast_json(contracts)


# ─────────────────── Lifespan Handler (Single, Corrected Version) ─────────────────── #

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DerivAPI + TradingStrategy singletons and tear them down cleanly."""
    app.state: State

    api_token = getattr(config, "DERIV_API_TOKEN", os.getenv("DERIV_API_TOKEN"))
    if not api_token:
        raise RuntimeError("DERIV_API_TOKEN missing – set in config or env var")

    app_id = 68054
    api_client = DerivAPI(app_id=app_id)

    app.state.deriv_api = api_client

    # CRITICAL FIX: Instantiate TradingStrategy WITH the callback.
    app.state.strategy = TradingStrategy(
        logger=logger,
        deriv_api_client=app.state.deriv_api,
        loginid=getattr(config, "LOGIN_ID", "Unknown"),
        portfolio_update_callback=broadcast_portfolio_updates,
    )
    app.state.streaming_task: asyncio.Task | None = None

    logger.info("Application ready. Waiting for user to start strategy.")
    yield

    # Shutdown logic
    logger.info("Initiating shutdown...")
    if hasattr(app.state, "streaming_task") and app.state.streaming_task and not app.state.streaming_task.done():
        app.state.streaming_task.cancel()

    if hasattr(app.state, "deriv_api") and app.state.deriv_api.has_connection:
        await app.state.deriv_api.disconnect()
    logger.info("Shutdown complete – API disconnected.")


# ────────────────────────── FastAPI App Initialization ──────────────────────────── #

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


# ────────────────────────── Pydantic Schemas ──────────────────────────── #

class StartRequest(BaseModel):
    symbols: List[str] | None = None


# ────────────────────────── Helper Functions for Endpoints ──────────────────────────── #

def _is_running() -> bool:
    task: asyncio.Task | None = app.state.streaming_task
    return task is not None and not task.done()


async def _run_strategy() -> None:
    """The main background task with detailed logging for the frontend."""
    strategy: TradingStrategy = app.state.strategy
    api_client: DerivAPI = app.state.deriv_api
    api_token = getattr(config, "DERIV_API_TOKEN", os.getenv("DERIV_API_TOKEN"))

    try:
        logger.info("Authorizing with Deriv API...")
        authorize_response = await api_client.authorize(api_token)
        if 'error' in authorize_response:
            logger.error(f"Authorization failed: {authorize_response['error']['message']}")
            return
        logger.info("Authorization successful.")

        if not strategy.active_symbols:
            logger.info(f"No symbols provided, using default: {config.SYMBOL}")
            strategy.set_active_symbols([config.SYMBOL])
        else:
            logger.info(f"Using provided symbols: {', '.join(strategy.active_symbols)}")

        await strategy.log_account_info()

        logger.info(f"Subscribing to ticks for {len(strategy.active_symbols)} symbol(s)...")
        await asyncio.gather(*(strategy.subscribe_live_ticks(s) for s in strategy.active_symbols))
        logger.info("All tick subscriptions are live. Strategy is now running.")

    except Exception as e:
        logger.error(f"An error occurred during strategy startup: {e}")
        raise


# ───────────────────────────── Endpoints ──────────────────────────────── #

@app.websocket("/ws/portfolio")
async def websocket_portfolio_endpoint(websocket: WebSocket):
    await portfolio_manager.connect(websocket)
    try:
        await broadcast_portfolio_updates()
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        portfolio_manager.disconnect(websocket)
        logger.info("Portfolio WS client disconnected.")


@app.get("/strategy/logs")
async def get_logs():
    return {"logs": list(log_capture)}


@app.post("/strategy/start")
async def start_strategy(body: StartRequest, bg: BackgroundTasks):
    if _is_running():
        raise HTTPException(status_code=400, detail="Strategy already running")

    log_capture.clear()
    strat: TradingStrategy = app.state.strategy
    if body.symbols:
        strat.set_active_symbols(body.symbols)

    task = asyncio.create_task(_run_strategy(), name="strategy-runner")
    app.state.streaming_task = task
    bg.add_task(lambda: task)
    return {"status": "starting", "symbols": strat.active_symbols}


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
        "running": _is_running(), "symbols": strat.active_symbols,
        "balance": strat.latest_balance, "wins": strat.total_wins,
        "losses": strat.total_losses, "net_pl": strat.cumulative_profit - strat.cumulative_loss,
        "drawdown": getattr(strat, "daily_pl", 0.0),
    }


@app.get("/strategy/open-contracts")
async def open_contracts():
    strat: TradingStrategy = app.state.strategy
    return await strat.get_open_contracts_info()


@app.post("/strategy/sell/{contract_id}")
async def sell_contract(contract_id: str):
    strat: TradingStrategy = app.state.strategy
    try:
        contract_id_int = int(contract_id)
        result = await strat.api_sell_contract(contract_id_int)
        await broadcast_portfolio_updates()
        return result
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
