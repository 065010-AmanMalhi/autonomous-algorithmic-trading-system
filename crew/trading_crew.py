"""
CrewAI multi-agent portfolio trader for Groq + local MCP servers.

Python/MCP fetches and computes market evidence. Three CrewAI specialists analyze
signals, build a portfolio plan, and produce a risk-constrained JSON decision set.
Python validates decisions and the independent MCP order server remains the final
authority for account, position, cash, frequency, halt, and paper-only controls.
"""
from __future__ import annotations

# Workaround: remove CrewAI's unsupported cache marker for Groq.
import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda message: message

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
            "macd_line": row.get("macd_line"),
            "macd_signal": row.get("macd_signal"),
            "macd_histogram": row.get("macd_histogram"),
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



def _deterministic_signal(
    row: dict,
    positions: dict,
    cfg: dict,
    account: dict,
    symbol: str,
    trading_halted: bool,
) -> dict:
    """Rule-based candidate gate; sizing and risk limits remain independent.

    Note: relative_volume in this compact workflow comes from a 1Day candle.
    Before the daily candle closes, its volume is partial but the denominator
    is the full-session 20-day average. Therefore it is diagnostic only here,
    never a hard entry confirmation.
    """
    if not isinstance(row, dict) or not row:
        return {"action": "hold", "reason": "market snapshot row missing", "buy_score": 0.0, "sell_score": 0.0}

    trend = str(row.get("trend", "")).strip().lower()
    macd = str(row.get("macd", "")).strip().lower()
    macd_hist_raw = row.get("macd_histogram")
    macd_histogram = _number(macd_hist_raw, -1e12)
    has_macd_histogram = macd_hist_raw is not None and abs(macd_histogram) < 1e11
    macd_line = _number(row.get("macd_line"), -1e12)
    macd_signal = _number(row.get("macd_signal"), -1e12)
    rsi = _number(row.get("rsi"), -1.0)
    bb = _number(row.get("bb_percent_b"), -999.0)
    relvol_raw = row.get("relative_volume")
    relvol = _number(relvol_raw, -1.0)
    held_value = max(0.0, _number((positions.get(symbol) or {}).get("market_value")))

    # Score trend/momentum/price location; don't score partial daily volume.
    buy_score = 0.0
    if trend == "up":
        buy_score += 2.0
    elif trend == "down":
        buy_score -= 2.0

    if macd == "bullish":
        buy_score += 2.0
    elif macd == "bearish":
        buy_score -= 2.0
    elif has_macd_histogram and macd_histogram > 0:
        # Histogram already positive without a fresh crossover: momentum is bullish.
        buy_score += 1.5
    elif has_macd_histogram and macd_histogram < 0:
        # Mild penalty for negative momentum, not equivalent to a fresh bearish cross.
        buy_score -= 0.5

    if 40.0 <= rsi <= 65.0:
        buy_score += 1.0
    elif 35.0 <= rsi < 40.0 or 65.0 < rsi <= 68.0:
        buy_score += 0.5
    elif 68.0 < rsi < 75.0:
        # Allow a modest RSI extension only when trend, MACD, and price location agree.
        buy_score -= 0.5
    elif rsi >= 75.0 or (0.0 < rsi < 30.0):
        buy_score -= 2.0

    if 0.2 <= bb <= 0.8:
        buy_score += 1.0
    elif 0.1 <= bb < 0.2 or 0.8 < bb <= 0.9:
        buy_score += 0.25
    elif bb < 0.0 or bb > 1.0:
        buy_score -= 1.0

    # RSI up to 75 is allowed for trend-following only if price is not
    # already overextended relative to the Bollinger bands.
    buy_rsi_ok = 35.0 <= rsi <= 75.0
    macd_bullish_momentum = (macd == "bullish") or (has_macd_histogram and macd_histogram > 0)
    buy_price_position_ok = 0.15 <= bb <= 0.85
    buy_confirmation = macd_bullish_momentum or buy_price_position_ok
    buy_eligible = (
        not trading_halted
        and trend == "up"
        and macd != "bearish"
        and buy_rsi_ok
        and buy_price_position_ok
        and buy_confirmation
        and buy_score >= 3.0
    )

    sell_score = 0.0
    if trend == "down":
        sell_score += 2.0
    elif trend == "up":
        sell_score -= 2.0
    if macd == "bearish":
        sell_score += 2.0
    elif macd == "bullish":
        sell_score -= 2.0
    elif has_macd_histogram and macd_histogram < 0:
        sell_score += 1.0
    elif has_macd_histogram and macd_histogram > 0:
        sell_score -= 0.5
    if rsi > 0 and rsi <= 40.0:
        sell_score += 1.0
    if 0.0 <= bb <= 0.2:
        sell_score += 1.0

    bearish_confirmation = (rsi > 0 and rsi <= 40.0) or (0.0 <= bb <= 0.2)
    sell_eligible = (
        held_value >= 1.0
        and (trend == "down" or macd == "bearish")
        and sell_score >= 3.0
        and bearish_confirmation
    )

    risk = cfg.get("risk", {})
    equity = _number(
        account.get("portfolio_value"),
        _number(account.get("equity"), cfg.get("portfolio", {}).get("starting_capital", 0.0)),
    )
    max_position = max(0.0, equity * _number(risk.get("max_position_weight_pct", 10.0)) / 100.0)
    position_headroom = max(0.0, max_position - held_value)

    diagnostics = (
        f"trend={row.get('trend', 'missing')}, RSI={rsi if rsi >= 0 else 'missing'}, "
        f"MACD crossover={row.get('macd', 'missing')}, "
        f"hist={round(macd_histogram, 6) if has_macd_histogram else 'missing'}, "
        f"line={round(macd_line, 6) if abs(macd_line) < 1e11 else 'missing'}, "
        f"signal={round(macd_signal, 6) if abs(macd_signal) < 1e11 else 'missing'}, "
        f"%B={round(bb, 3) if -10 < bb < 10 else 'missing'}, "
        f"relative_volume={relvol if relvol >= 0 else 'missing'} (diagnostic only for 1Day bars)"
    )

    if buy_eligible and position_headroom >= 1.0:
        # Starter tranche = 25% of the per-symbol position cap (3% of equity
        # at the current 12% cap), subject to central cash/position validation.
        notional = min(max_position * 0.25, position_headroom)
        if notional >= 1.0:
            confidence = min(0.85, 0.55 + 0.04 * max(0.0, buy_score - 3.0))
            return {
                "action": "buy", "notional": round(notional, 2),
                "confidence": round(confidence, 3), "buy_score": round(buy_score, 1),
                "sell_score": round(sell_score, 1),
                "reason": f"uptrend with momentum/price confirmation; {diagnostics}",
            }

    if sell_eligible:
        confidence = min(0.85, 0.55 + 0.04 * max(0.0, sell_score - 3.0))
        return {
            "action": "sell", "notional": round(held_value, 2),
            "confidence": round(confidence, 3), "buy_score": round(buy_score, 1),
            "sell_score": round(sell_score, 1),
            "reason": f"held position with confirmed bearish setup; {diagnostics}",
        }

    why = []
    if trading_halted:
        why.append("trading halt active")
    if trend != "up":
        why.append(f"trend={row.get('trend', 'missing')}")
    if not buy_rsi_ok:
        why.append(f"RSI={rsi if rsi >= 0 else 'missing'} outside 35-75")
    if macd == "bearish":
        why.append("MACD crossover bearish")
    if not buy_price_position_ok:
        why.append(f"Bollinger %B={bb if -10 < bb < 10 else 'missing'} outside 0.15-0.85 entry range")
    if not buy_confirmation:
        why.append("no positive MACD histogram/crossover or constructive Bollinger position")
    if buy_score < 3.0:
        why.append(f"buy score {buy_score:.1f} below 3.0")
    if held_value < 1.0:
        why.append("no held position to exit")
    why.append(diagnostics)
    return {
        "action": "hold", "buy_score": round(buy_score, 1),
        "sell_score": round(sell_score, 1), "reason": "; ".join(why),
    }
def _normalize_decisions(
    model_output: str,
    symbols: list[str],
    cfg: dict,
    account: dict,
    positions: dict,
    trading_halted: bool = False,
    snapshot: dict | None = None,
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

    # If the model says HOLD, independently check explicit technical entry/exit rules.
    # A rule-based action is only proposed when measured fields meet the criteria; no forced trades.
    rows_by_symbol = {
        str(row.get("symbol", "")).upper(): row
        for row in (snapshot or {}).get("rows", [])
        if isinstance(row, dict)
    }
    for symbol in symbols:
        model_item = by_symbol.get(symbol, {})
        if str(model_item.get("action", "hold")).lower().strip() != "hold":
            continue
        signal = _deterministic_signal(
            rows_by_symbol.get(symbol, {}), positions, cfg, account, symbol, trading_halted
        )
        signal_action = signal.get("action", "hold")
        if signal_action in {"buy", "sell"}:
            by_symbol[symbol] = {
                "symbol": symbol,
                "action": signal_action,
                "notional": signal.get("notional", 0.0),
                "confidence": signal.get("confidence", 0.0),
                "reason": signal.get("reason", "deterministic rule signal"),
            }
            notes.append(
                f"{symbol}: model HOLD overridden by rule-based {signal_action.upper()} "
                f"(buy_score={signal.get('buy_score', 0)}, "
                f"sell_score={signal.get('sell_score', 0)}; {signal.get('reason', '')})."
            )
        else:
            notes.append(
                f"{symbol}: rule HOLD (buy_score={signal.get('buy_score', 0)}, "
                f"sell_score={signal.get('sell_score', 0)}; {signal.get('reason', '')})."
            )

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
        "Evaluate each symbol individually for an Alpaca PAPER account using only the supplied "
        "market snapshot, positions, account, and limits. Trend is 'Up', 'Down', or 'Flat'. "
        "The macd field is a crossover category: 'Bullish', 'Bearish', or 'None'; 'None' means "
        "no fresh crossover, not that momentum is missing. macd_line, macd_signal, and "
        "macd_histogram are numeric; a positive histogram means MACD is above its signal line. "
        "RSI is numeric and bb_percent_b is a Bollinger position "
        "value normally between 0 and 1. relative_volume is diagnostic only: because the compact "
        "workflow uses 1Day bars, the current bar's volume may be partial during market hours and "
        "must not be treated as a full-session relative-volume reading.\n\n"
        "ENTRY RULE: actively evaluate a small starter BUY when trend is 'Up', MACD histogram "
        "is positive (or crossover is 'Bullish'), RSI is 35-75, and bb_percent_b is 0.15-0.85. "
        "RSI 69-75 is allowed only when trend and momentum are positive and price remains below "
        "the upper Bollinger zone; do not chase when %B > 0.85. Favor HOLD if histogram is "
        "non-positive without a bullish crossover, price location is stretched, the trend is not Up, "
        "required data is missing, or trading is halted. A fresh crossover is not required when "
        "the histogram is already positive. "
        "EXIT RULE: consider SELL only for an existing long position when trend is Down or MACD is "
        "Bearish, and RSI <= 40 or bb_percent_b <= 0.2 confirms weakness. Never short, use leverage, "
        "or propose options.\n\n"
        "Confidence must reflect the evidence; buys must meet the configured confidence threshold. "
        "Never invent data. Python and the paper-order server independently enforce position size, "
        "cash buffer, daily trade frequency, and daily loss halt. Return only valid JSON with a "
        "'decisions' list containing exactly one entry per supplied symbol. Each entry must include "
        "symbol, action (buy/sell/hold), notional (zero for HOLD and positive for BUY), confidence "
        "(0-1), and reason (at most 8 words). Do not copy examples or output markdown.\n\n"
        "DATA:\n" + json.dumps(context, separators=(",", ":"), ensure_ascii=True)
    )
    return description



def _validate_model_output(raw: str, symbols: list[str]) -> None:
    parsed = _extract_json_object(raw)
    rows = parsed["decisions"]
    if len(rows) != len(symbols):
        raise ValueError(f"Expected {len(symbols)} decisions, received {len(rows)}")
    by_symbol = {}
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("A decision entry is not an object")
        symbol = str(item.get("symbol", "")).upper().strip()
        if symbol not in symbols or symbol in by_symbol:
            raise ValueError(f"Unknown or duplicate symbol: {symbol!r}")
        action = str(item.get("action", "")).lower().strip()
        confidence = _number(item.get("confidence"), -1)
        notional = _number(item.get("notional"), -1)
        if action not in {"buy", "sell", "hold"}:
            raise ValueError(f"{symbol}: invalid action")
        if not 0 <= confidence <= 1:
            raise ValueError(f"{symbol}: invalid confidence")
        if notional < 0:
            raise ValueError(f"{symbol}: invalid notional")
        if action == "buy" and notional <= 0:
            raise ValueError(f"{symbol}: BUY requires positive notional")
        if action == "hold" and notional != 0:
            raise ValueError(f"{symbol}: HOLD must have zero notional")
        by_symbol[symbol] = item
    if set(by_symbol) != set(symbols):
        raise ValueError("Response is missing portfolio symbols")
    if all(
        str(item.get("action", "")).lower().strip() == "hold"
        and _number(item.get("confidence"), -1) == 0.5
        and _number(item.get("notional"), -1) == 0
        and str(item.get("reason", "")).strip().lower() == "mixed signals"
        for item in rows
    ):
        raise ValueError("Detected repeated example-style HOLD response")


def _run_decision_with_fallback(
    cfg: dict, task_description: str, symbols: list[str], api_key: str
) -> tuple[str, str, list[str]]:
    llm_cfg = cfg["llm"]
    candidates = [llm_cfg.get("model"), *llm_cfg.get("fallback_models", [])]
    models = list(dict.fromkeys(
        model.strip() for model in candidates
        if isinstance(model, str) and model.strip()
    ))
    if not models:
        raise RuntimeError("No LLM models are configured")
    errors = []
    for model_name in models:
        try:
            llm = LLM(
                model=model_name, api_key=api_key,
                temperature=float(llm_cfg.get("temperature", 0.2)),
                max_tokens=int(llm_cfg.get("max_tokens", 1800)),
            )
            # Three real CrewAI specialists run sequentially on every successful cycle.
            # All agents see evidence only; Python/order-server safeguards remain authoritative.
            analyst = Agent(
                role="Technical Signal Analyst",
                goal="Identify evidence-backed trend and momentum candidates for every symbol.",
                backstory=(
                    "You are a technical analyst. Read numeric RSI, MACD histogram/lines, "
                    "crossover, trend, and Bollinger %B carefully. None crossover means no "
                    "fresh crossover, not missing momentum. Never invent market data."
                ),
                llm=llm, verbose=False, max_iter=1, allow_delegation=False,
            )
            strategist = Agent(
                role="Portfolio Strategist",
                goal="Turn technical evidence into balanced BUY, SELL, and HOLD proposals.",
                backstory=(
                    "You manage a diversified long-only paper portfolio. You seek valid "
                    "opportunities without forcing trades, chasing extended prices, shorting, "
                    "using leverage, or proposing options."
                ),
                llm=llm, verbose=False, max_iter=1, allow_delegation=False,
            )
            risk_supervisor = Agent(
                role="Risk & Execution Supervisor",
                goal="Produce the final schema-valid decision set within every configured limit.",
                backstory=(
                    "You are the final pre-trade reviewer. Never short or use leverage. "
                    "Use exact positions/cash and leave the reserve intact. Halt means no buys. "
                    "A SELL requires an existing long position."
                ),
                llm=llm, verbose=False, max_iter=1, allow_delegation=False,
            )

            analysis_task = Task(
                description=(
                    "Analyze all symbols in the supplied original market/account context. "
                    "For every symbol, report trend, RSI, MACD crossover, MACD histogram, "
                    "Bollinger %B, and classify the setup as bullish, mixed, or bearish. "
                    "Explicitly identify whether positive histogram momentum persists without "
                    "a fresh crossover. Identify possible entries only when trend is Up, "
                    "RSI is 35-75, histogram is positive or crossover is Bullish, and %B is "
                    "0.15-0.85. Keep it concise and numeric; do not size orders or invent data.\n\n"
                    + task_description
                ),
                expected_output="Concise per-symbol technical assessment using supplied numbers only.",
                agent=analyst,
            )
            strategy_task = Task(
                description=(
                    "Use the technical analyst's findings to assess every symbol. Recommend "
                    "BUY only for an Up trend, RSI 35-75, positive MACD histogram or Bullish "
                    "crossover, and Bollinger %B 0.15-0.85. RSI 69-75 is not an automatic veto "
                    "when momentum is positive and price remains in that band. Avoid chasing %B "
                    "above 0.85. SELL only an existing long when trend is Down or MACD is Bearish "
                    "and RSI <= 40 or %B <= 0.20 confirms weakness. Otherwise HOLD. Consider all "
                    "eligible candidates; do not default the entire portfolio to HOLD without "
                    "evaluating each row. Return a concise symbol/action/reason proposal, not JSON."
                ),
                expected_output="Concise proposed portfolio actions with an evidence-based reason per candidate.",
                agent=strategist,
                context=[analysis_task],
            )
            decision_task = Task(
                description=(
                    "Finalize the entire portfolio from the original evidence and preceding "
                    "specialist outputs. Apply the stated rules; do not force a trade where no "
                    "candidate qualifies, but do not leave eligible candidates at HOLD merely "
                    "because RSI is between 69 and 75 when positive momentum and %B 0.15-0.85 "
                    "agree. Return only valid JSON with exactly one entry per supplied symbol, "
                    "keys symbol/action/notional/confidence/reason, and actions buy/sell/hold. "
                    "HOLD notional must be 0. Buy notional must be positive and within the "
                    "max_position_dollars and cash reserve; use a starter size near 25% of the "
                    "per-position cap where account capacity permits. Confidence must be 0-1 and "
                    "meet the configured minimum for BUY. Never buy during a halt. Never short, "
                    "use leverage, invent data, or propose options. Limit each reason to 8 words.\n\n"
                    "ORIGINAL EVIDENCE AND LIMITS:\n" + task_description
                ),
                expected_output="Only valid JSON with one risk-reviewed decision per portfolio symbol.",
                agent=risk_supervisor,
                context=[analysis_task, strategy_task],
            )
            crew = Crew(
                agents=[analyst, strategist, risk_supervisor],
                tasks=[analysis_task, strategy_task, decision_task],
                process=Process.sequential,
                verbose=False,
            )
            print("  CrewAI pipeline: Technical Signal Analyst -> Portfolio Strategist -> Risk & Execution Supervisor")
            raw = str(crew.kickoff())
            # Keep each specialist's actual CrewAI output for the assignment audit trail.
            for stage_name, stage_task in (
                ("TechnicalSignalAnalyst", analysis_task),
                ("PortfolioStrategist", strategy_task),
                ("RiskExecutionSupervisor", decision_task),
            ):
                stage_output = getattr(stage_task, "output", None)
                if stage_output is not None:
                    stage_text = getattr(stage_output, "raw", None) or str(stage_output)
                    try:
                        save_agent_log(cfg.get("logs", {}).get("run_dir", "logs"), stage_name, str(stage_text))
                    except Exception as log_exc:
                        print(f"  [CrewAI log warning] {stage_name}: {type(log_exc).__name__}: {log_exc}")
            _validate_model_output(raw, symbols)
            if errors:
                print(
                    f"  Model fallback succeeded: {model_name} "
                    f"after {len(errors)} failed attempt(s)."
                )
            return raw, model_name, errors
        except Exception as exc:
            message = f"{type(exc).__name__}: {str(exc)[:220]}"
            errors.append(f"{model_name} -> {message}")
            print(f"  [LLM fallback] {model_name} failed: {message}")
    raise RuntimeError(
        "All configured LLM models failed validation/access; no orders submitted. "
        + " | ".join(errors)
    )

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
        task_description = _decision_task_description(
            cfg, snapshot, account, positions, trading_halted
        )
        raw_decision, selected_model, model_attempt_errors = (
            _run_decision_with_fallback(
                cfg=cfg, task_description=task_description,
                symbols=symbols, api_key=api_key,
            )
        )

        decisions, validation_notes = _normalize_decisions(
            raw_decision, symbols, cfg, account, positions, trading_halted, snapshot=snapshot
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
                "selected_model": selected_model,
                "model_attempt_errors": model_attempt_errors,
                "pre_order_validation_notes": validation_notes,
                "normalized_decisions": decisions,
                "execution": execution_result,
            }, indent=2, default=str),
        )
        print(
            f"\n{'='*64}\n"
            f"  {cfg.get('student_name', 'Portfolio')} | {len(symbols)} symbols\n"
            f"  LLM attempts: {len(model_attempt_errors) + 1} | "
            f"selected model: {selected_model} | Alpaca PAPER only\n"
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
