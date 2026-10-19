"""
Low-token, single-inference portfolio trader for Groq + local MCP servers.

Data fetching and indicator calculations run deterministically in Python/MCP.
Groq receives one compact portfolio snapshot and returns one JSON decision set.
Python validates that output and submits one batch to Alpaca PAPER trading.
The independent order server remains the final authority for risk checks.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

import yaml
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logging_utils import save_agent_log

SERVERS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "mcp_servers", "stock_trading", "servers",
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _tool_name(tool: Any) -> str:
    name = getattr(tool, "name", None) or getattr(tool, "tool_name", None) or ""
    return str(name).rsplit("__", 1)[-1].rsplit("/", 1)[-1]


def tools_for(all_tools: list, *wanted_names: str) -> dict[str, Any]:
    """Find only the expected tools, including server-prefixed MCP names."""
    found = {}
    names = [(tool, _tool_name(tool)) for tool in all_tools]
    for wanted in wanted_names:
        matches = [tool for tool, name in names if name == wanted]
        if not matches:
            matches = [
                tool for tool, name in names
                if name.endswith("_" + wanted) or name.endswith("__" + wanted)
            ]
        if len(matches) != 1:
            available = sorted(name for _, name in names)
            raise RuntimeError(
                f"Expected one MCP tool named {wanted!r}; found {len(matches)}. "
                f"Available tools: {available}"
            )
        found[wanted] = matches[0]
    return found


def _invoke_tool(tool: Any, arguments: dict) -> Any:
    """Call a CrewAI MCP tool directly without spending an LLM turn."""
    # CrewAIToolAdapter creates BaseTool subclasses whose _run signature is
    # keyword-only, so use run(**arguments), not run(arguments).
    if hasattr(tool, "run"):
        return tool.run(**arguments)
    if hasattr(tool, "invoke"):
        return tool.invoke(arguments)
    raise TypeError(f"MCP tool {_tool_name(tool)!r} has no run/invoke method")


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    # MCP wrappers may return an object with textual content blocks.
    content = getattr(value, "content", None)
    if content:
        for item in content:
            item_text = getattr(item, "text", None)
            if item_text:
                try:
                    parsed = json.loads(item_text)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue
    return {"raw": str(value)}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if result == result and abs(result) != float("inf") else default
    except (TypeError, ValueError):
        return default


def _extract_json_object(raw: str) -> dict:
    """Accept plain JSON or a fenced JSON object, but never guess a decision."""
    text = str(raw).strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("decisions"), list):
            return value
    raise ValueError("Groq did not return a valid JSON object with a decisions list")


def _compact_snapshot(snapshot: dict) -> dict:
    """Whitelist fields so unexpected tool payloads cannot inflate the prompt."""
    rows = []
    for row in snapshot.get("rows", []):
        if not isinstance(row, dict):
            continue
        rows.append({
            "symbol": str(row.get("symbol", "")).upper(),
            "close": row.get("close"),
            "rsi": row.get("rsi"),
            "rsi_signal": row.get("rsi_signal"),
            "macd": row.get("macd"),
            "bb_percent_b": row.get("bb_percent_b"),
            "trend": row.get("trend"),
            "relative_volume": row.get("relative_volume"),
            "beta": row.get("beta"),
            "beta_class": row.get("beta_class"),
        })
    return {
        "status": snapshot.get("status", "unknown"),
        "equal_weight_average_beta": snapshot.get("equal_weight_average_beta"),
        "rows": rows,
        "missing": [
            {"symbol": str(item.get("symbol", "")).upper(),
             "error": str(item.get("error", ""))[:100]}
            for item in snapshot.get("missing", [])[:15]
            if isinstance(item, dict)
        ],
    }


def _compact_positions(positions: dict, allowed_symbols: list[str]) -> dict:
    allowed = set(allowed_symbols)
    compact = {}
    for position in positions.get("positions", []):
        if not isinstance(position, dict):
            continue
        symbol = str(position.get("symbol", "")).upper()
        if symbol in allowed:
            compact[symbol] = {
                "qty": _number(position.get("qty")),
                "market_value": round(_number(position.get("market_value")), 2),
            }
    return compact


def _normalize_decisions(
    model_output: str,
    symbols: list[str],
    cfg: dict,
    account: dict,
    positions: dict,
    trading_halted: bool = False,
) -> tuple[list[dict], list[str]]:
    """
    Fail closed on invalid output, unknown symbols and malformed trades.
    Locally applies position and cash caps before the server's own checks.
    """
    notes: list[str] = []
    try:
        parsed = _extract_json_object(model_output)
    except ValueError as exc:
        notes.append(str(exc) + "; submitted HOLD for all symbols.")
        return [
            {"symbol": symbol, "action": "hold", "notional": 0.0, "confidence": 0.0}
            for symbol in symbols
        ], notes

    by_symbol = {}
    for item in parsed["decisions"]:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).upper().strip()
        if symbol not in symbols or symbol in by_symbol:
            notes.append(f"Ignored unknown/duplicate symbol: {symbol or '(blank)'}")
            continue
        by_symbol[symbol] = item

    limits = cfg.get("risk", {})
    equity = _number(account.get("portfolio_value"), _number(account.get("equity"), cfg["portfolio"]["starting_capital"]))
    cash = _number(account.get("cash"))
    position_cap = max(0.0, equity * _number(limits.get("max_position_weight_pct", 10)) / 100)
    cash_floor = max(0.0, equity * _number(limits.get("min_cash_buffer_pct", 10)) / 100)
    buy_capacity = max(0.0, cash - cash_floor)
    confidence_floor = _number(limits.get("min_confidence_to_trade", 0.65))
    # The server independently checks its on-disk daily trade ledger every call.

    normalized = []
    positions_by_symbol = positions
    for symbol in symbols:
        item = by_symbol.get(symbol, {})
        action = str(item.get("action", "hold")).lower().strip()
        confidence = _number(item.get("confidence"), 0.0)
        notional = _number(item.get("notional"), 0.0)
        if action not in {"buy", "sell", "hold"}:
            notes.append(f"{symbol}: invalid action converted to hold")
            action = "hold"
        if confidence < 0 or confidence > 1:
            notes.append(f"{symbol}: invalid confidence converted to hold")
            action = "hold"
        if action == "buy":
            if trading_halted:
                notes.append(f"{symbol}: trading halt active; buy converted to hold")
                action = "hold"
            elif confidence < confidence_floor:
                notes.append(f"{symbol}: buy below confidence threshold converted to hold")
                action = "hold"
            elif notional <= 0:
                notes.append(f"{symbol}: buy with invalid notional converted to hold")
                action = "hold"
            else:
                existing_value = max(0.0, _number(positions_by_symbol.get(symbol, {}).get("market_value")))
                position_headroom = max(0.0, position_cap - existing_value)
                proposed = notional
                notional = min(notional, position_headroom, buy_capacity)
                if notional < proposed:
                    notes.append(f"{symbol}: buy size reduced to position/cash headroom")
                if notional < 1.0:
                    notes.append(f"{symbol}: insufficient cash after buffer; hold")
                    action = "hold"
                else:
                    buy_capacity -= notional
        elif action == "sell":
            held = positions_by_symbol.get(symbol, {})
            available_market_value = max(0.0, _number(held.get("market_value")))
            if available_market_value < 1.0:
                notes.append(f"{symbol}: sell without a long position converted to hold")
                action = "hold"
            elif notional <= 0:
                # Default to closing the long position; selling cannot exceed holdings.
                notional = available_market_value
            else:
                notional = min(notional, available_market_value)
            if notional < 1.0:
                action = "hold"
        if action == "hold":
            notional = 0.0
        normalized.append({
            "symbol": symbol,
            "action": action,
            "notional": round(notional, 2),
            "confidence": round(confidence, 3),
        })
    return normalized, notes


def _decision_task_description(cfg: dict, snapshot: dict, account: dict, positions: dict, trading_halted: bool = False) -> str:
    symbols = [str(item["symbol"]).upper() for item in cfg["portfolio"]["stocks"]]
    risk = cfg["risk"]
    portfolio_value = _number(account.get("portfolio_value"), _number(account.get("equity"), cfg["portfolio"]["starting_capital"]))
    cash = _number(account.get("cash"))
    max_position = portfolio_value * _number(risk.get("max_position_weight_pct", 10)) / 100
    cash_floor = portfolio_value * _number(risk.get("min_cash_buffer_pct", 10)) / 100
    context = {
        "symbols": symbols,
        "account": {"portfolio_value": round(portfolio_value, 2), "cash": round(cash, 2)},
        "positions": positions,
        "limits": {
            "max_position_dollars": round(max_position, 2),
            "minimum_cash_dollars": round(cash_floor, 2),
            "min_confidence_to_trade": risk["min_confidence_to_trade"],
            "max_trades_per_symbol_per_day": risk["max_trades_per_symbol_per_day"],
        },
        "market_snapshot": snapshot,
        "trading_halted": bool(trading_halted),
    }
    description = (
        "You are making one conservative decision pass for an Alpaca PAPER account. "
        "Use only the supplied snapshot and current positions. Do not invent prices or data. "
        "Choose BUY only with a coherent bullish setup (up trend, bullish MACD crossover, "
        "RSI generally 35-68, and no obvious contradiction). Choose SELL only to reduce an "
        "existing long position when the setup is bearish (down trend or bearish MACD). "
        "Otherwise choose HOLD. Never short, never propose options, and never use leverage. "
        "Use confidence from 0 to 1. BUY confidence must meet the configured threshold. "
        "Each order notional must be positive and no greater than max_position_dollars; total "
        "buys must leave the minimum cash reserve intact. Python and the paper-order server "
        "will enforce these limits again. Prefer HOLD when signals or data are missing. "
        "Limit each reason to 8 words. Return only valid JSON (no markdown, no extra text) "
        "in this exact shape: {\"decisions\":[{\"symbol\":\"AAPL\",\"action\":\"hold\","
        "\"notional\":0,\"confidence\":0.5,\"reason\":\"mixed signals\"}]}. Include exactly "
        "one entry for every supplied symbol; allowed actions are buy, sell, hold.\n\n"
        "DATA:\n" + json.dumps(context, separators=(",", ":"), ensure_ascii=True)
    )
    return description


def run_one_cycle(config_path: str):
    cfg = load_config(config_path)
    symbols = [str(item["symbol"]).upper() for item in cfg["portfolio"]["stocks"]]
    log_dir = cfg["logs"]["run_dir"]
    server_env = {
        **os.environ,
        "TRADING_CONFIG_PATH": os.path.abspath(config_path),
        "TRADING_LOG_DIR": os.path.abspath(log_dir),
    }
    indicators_server = StdioServerParameters(
        command=sys.executable,
        args=[os.path.join(SERVERS_DIR, "technical_indicators_mcp_server.py")],
        env=server_env,
    )
    orders_server = StdioServerParameters(
        command=sys.executable,
        args=[os.path.join(SERVERS_DIR, "alpaca_orders_mcp_server.py")],
        env=server_env,
    )

    with MCPServerAdapter([indicators_server, orders_server]) as mcp_tools:
        selected = tools_for(
            mcp_tools,
            "fetch_ohlcv_batch",
            "calculate_portfolio_snapshot",
            "get_all_positions",
            "get_account_info",
            "get_safety_status",
            "place_portfolio_orders",
        )
        data_cfg = cfg["data"]
        # Data calls are direct MCP invocations, not LLM tool calls.
        fetch_result = _as_dict(_invoke_tool(selected["fetch_ohlcv_batch"], {
            "symbols": symbols,
            "lookback_days": int(data_cfg.get("lookback_days", 120)),
            "timeframe": data_cfg.get("timeframe", "1Day"),
            "include_benchmark": True,
            "include_market_fear_proxy": False,
        }))
        if fetch_result.get("status") not in {"ok", "partial"}:
            raise RuntimeError(f"OHLCV fetch failed; refusing to ask the model to trade: {fetch_result}")
        snapshot_raw = _as_dict(_invoke_tool(selected["calculate_portfolio_snapshot"], {
            "symbols": symbols,
            "beta_window_days": int(data_cfg.get("beta_window_days", 60)),
        }))
        snapshot = _compact_snapshot(snapshot_raw)
        account = _as_dict(_invoke_tool(selected["get_account_info"], {}))
        positions_raw = _as_dict(_invoke_tool(selected["get_all_positions"], {}))
        safety = _as_dict(_invoke_tool(selected["get_safety_status"], {}))
        if "raw" in account or "cash" not in account or not ("portfolio_value" in account or "equity" in account):
            raise RuntimeError("Could not verify current paper-account cash/equity; refusing to trade.")
        if "raw" in positions_raw or not isinstance(positions_raw.get("positions"), list):
            raise RuntimeError("Could not verify open positions; refusing to trade.")
        positions = _compact_positions(positions_raw, symbols)
        if account.get("account_blocked") or account.get("trading_blocked"):
            raise RuntimeError("Alpaca reports the paper account is blocked; no decisions or orders were submitted.")

        trading_halted = bool(safety.get("halted", True))
        llm_cfg = cfg["llm"]
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add your Groq key to .env.")
        llm = LLM(
            model=llm_cfg["model"],
            api_key=api_key,
            temperature=float(llm_cfg.get("temperature", 0.1)),
            max_tokens=int(llm_cfg.get("max_tokens", 1200)),
        )
        decision_agent = Agent(
            role="Compact Portfolio Decision-Maker",
            goal="Return a single compact JSON portfolio decision set based on provided numeric evidence.",
            backstory="You are conservative, evidence-based and operate on an Alpaca PAPER account. Missing or contradictory data means HOLD.",
            llm=llm,
            verbose=False,
            max_iter=1,
            allow_delegation=False,
        )
        task = Task(
            description=_decision_task_description(cfg, snapshot, account, positions, trading_halted),
            expected_output="Only valid JSON with one buy/sell/hold decision per portfolio symbol.",
            agent=decision_agent,
        )
        crew = Crew(agents=[decision_agent], tasks=[task], process=Process.sequential, verbose=False)
        raw_decision = str(crew.kickoff())

        decisions, validation_notes = _normalize_decisions(
            raw_decision, symbols, cfg, account, positions, trading_halted
        )
        # The order server independently enforces account, position, cash,
        # trade-frequency, halt, and other configured limits.
        execution_result = _as_dict(_invoke_tool(
            selected["place_portfolio_orders"],
            {"decisions": decisions},
        ))

        save_agent_log(log_dir, "PortfolioDataFetcher", json.dumps(fetch_result, indent=2))
        save_agent_log(log_dir, "PortfolioTechnicalAnalyst", json.dumps(snapshot, indent=2))
        save_agent_log(log_dir, "PortfolioDecisionMaker", raw_decision)
        save_agent_log(
            log_dir,
            "PortfolioExecutionAgent",
            json.dumps({
                "safety_before_order": safety,
                "pre_order_validation_notes": validation_notes,
                "normalized_decisions": decisions,
                "execution": execution_result,
            }, indent=2, default=str),
        )
        print(
            f"\n{'='*64}\n"
            f"  {cfg.get('student_name', 'Portfolio')} | {len(symbols)} symbols\n"
            f"  LLM calls: 1 | model: {llm_cfg['model']} | Alpaca PAPER only\n"
            f"  Options/regression: disabled in compact workflow\n"
            f"  Safety halt active before order: {bool(safety.get('halted', False))}\n"
            f"{'='*64}\n"
        )
        print(json.dumps({
            "decisions": decisions,
            "validation_notes": validation_notes,
            "execution": execution_result,
        }, indent=2, default=str))
        return execution_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/config.example.yaml")
    args = parser.parse_args()
    run_one_cycle(args.config)
