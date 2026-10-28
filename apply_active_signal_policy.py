from __future__ import annotations

import ast
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
CREW = ROOT / "crew" / "trading_crew.py"

RULE_HELPER = r'''
def _deterministic_signal(
    row: dict,
    positions: dict,
    cfg: dict,
    account: dict,
    symbol: str,
    trading_halted: bool,
) -> dict:
    """Transparent technical entry/exit gate used when the model says HOLD.

    This does not bypass risk limits. BUY candidates are still processed by
    _normalize_decisions and the independent Alpaca order server.
    """
    if not isinstance(row, dict) or not row:
        return {"action": "hold", "reason": "market snapshot row missing", "buy_score": 0.0, "sell_score": 0.0}

    trend = str(row.get("trend", "")).strip().lower()
    macd = str(row.get("macd", "")).strip().lower()
    rsi = _number(row.get("rsi"), -1.0)
    bb = _number(row.get("bb_percent_b"), -999.0)
    relvol = _number(row.get("relative_volume"), -1.0)
    held_value = max(0.0, _number((positions.get(symbol) or {}).get("market_value")))

    # MACD is a categorical *crossover event* in this snapshot, not a numeric line.
    buy_score = 0.0
    if trend == "up":
        buy_score += 2.0
    elif trend == "down":
        buy_score -= 2.0
    if macd == "bullish":
        buy_score += 2.0
    elif macd == "bearish":
        buy_score -= 2.0
    if 40.0 <= rsi <= 65.0:
        buy_score += 1.0
    elif 35.0 <= rsi < 40.0 or 65.0 < rsi <= 68.0:
        buy_score += 0.5
    elif rsi >= 70.0 or (0.0 < rsi < 30.0):
        buy_score -= 2.0
    if 0.2 <= bb <= 0.8:
        buy_score += 1.0
    elif bb < 0.0 or bb > 1.0:
        buy_score -= 1.0
    if relvol >= 0.9:
        buy_score += 1.0
    elif 0.0 <= relvol < 0.5:
        buy_score -= 1.0

    bearish_confirmation = (rsi > 0 and rsi <= 40.0) or (0.0 <= bb <= 0.2)
    buy_rsi_ok = 35.0 <= rsi <= 68.0
    buy_confirmation = (macd == "bullish") or (0.2 <= bb <= 0.8 and relvol >= 0.9)
    buy_eligible = (
        not trading_halted
        and trend == "up"
        and macd != "bearish"
        and buy_rsi_ok
        and buy_confirmation
        and buy_score >= 4.5
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
    if rsi > 0 and rsi <= 40.0:
        sell_score += 1.0
    if 0.0 <= bb <= 0.2:
        sell_score += 1.0
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
    existing_value = held_value
    position_headroom = max(0.0, max_position - existing_value)

    if buy_eligible and position_headroom >= 1.0:
        # A starter tranche is 25% of the configured per-symbol position cap.
        notional = min(max_position * 0.25, position_headroom)
        if notional >= 1.0:
            confidence = min(0.85, 0.55 + 0.04 * max(0.0, buy_score - 4.0))
            return {
                "action": "buy", "notional": round(notional, 2),
                "confidence": round(confidence, 3), "buy_score": round(buy_score, 1),
                "sell_score": round(sell_score, 1),
                "reason": "uptrend with momentum/volume confirmation",
            }

    if sell_eligible:
        confidence = min(0.85, 0.55 + 0.04 * max(0.0, sell_score - 3.0))
        return {
            "action": "sell", "notional": round(held_value, 2),
            "confidence": round(confidence, 3), "buy_score": round(buy_score, 1),
            "sell_score": round(sell_score, 1),
            "reason": "held position with confirmed bearish setup",
        }

    why = []
    if trend != "up":
        why.append(f"trend={row.get('trend', 'missing')}")
    if not buy_rsi_ok:
        why.append(f"RSI={rsi if rsi >= 0 else 'missing'} outside 35-68")
    if macd == "bearish":
        why.append("MACD crossover bearish")
    if not buy_confirmation:
        why.append("no MACD/BB-volume confirmation")
    if held_value < 1.0:
        why.append("no held position to exit")
    return {
        "action": "hold", "buy_score": round(buy_score, 1),
        "sell_score": round(sell_score, 1),
        "reason": "; ".join(why) if why else "no rule threshold met",
    }
'''

NEW_DESCRIPTION = r'''    description = (
        "Evaluate every symbol individually for an Alpaca PAPER account. Use only the supplied "
        "snapshot, positions, account, and limits. The snapshot's trend values are exactly "
        "'Up', 'Down', or 'Flat'. Its macd field is a categorical crossover event: 'Bullish', "
        "'Bearish', or 'None'—it is NOT a numeric MACD line. 'None' means there was no fresh "
        "histogram crossover on the latest bar; it does not mean bearish. RSI is numeric, "
        "bb_percent_b is usually 0-1, and relative_volume is a numeric ratio.\n\n"
        "ENTRY RULE: consider BUY only if trend is 'Up', RSI is 35-68, MACD is not 'Bearish', "
        "and there is confirmation from either MACD='Bullish' OR (bb_percent_b between 0.2 "
        "and 0.8 AND relative_volume >= 0.9). If those conditions are met, do not default to "
        "HOLD merely because the MACD field says 'None'; weigh the actual supporting data. "
        "EXIT RULE: consider SELL only for a symbol with an existing long position and a "
        "bearish trend or bearish MACD crossover plus confirmation from RSI <= 40 or "
        "bb_percent_b <= 0.2. Otherwise HOLD. Never short, use leverage, or propose options.\n\n"
        "Confidence must reflect the supplied evidence. A BUY must meet the configured "
        "min_confidence_to_trade threshold. Never invent data. The Python validation layer and "
        "the order server enforce position size, cash buffer, daily trade frequency, and daily "
        "loss halt independently. Total buy orders must leave the minimum cash reserve intact. "
        "Return only valid JSON with one key, 'decisions', containing exactly one entry per "
        "supplied symbol. Each entry must contain symbol, action (buy/sell/hold), notional "
        "(non-negative number; zero for HOLD; positive for BUY), confidence (0-1), and reason "
        "(at most 8 words). Do not copy an example or output markdown.\n\n"
        "DATA:\n" + json.dumps(context, separators=(",", ":"), ensure_ascii=True)
    )
    return description'''


def backup(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = path.with_suffix(path.suffix + f".{stamp}.bak")
    shutil.copy2(path, target)
    print(f"Backup created: {target}")
    return target


def patch_source(source: str) -> str:
    changed = source

    # Replace the prompt safely via a callable regex replacement (avoid \n being expanded).
    prompt_start = changed.find("def _decision_task_description(")
    prompt_end = changed.find("\ndef run_one_cycle(", prompt_start)
    if prompt_start < 0 or prompt_end < 0:
        raise RuntimeError("Could not find _decision_task_description function")
    prompt_section = changed[prompt_start:prompt_end]
    new_section, count = re.subn(
        r"(?ms)^    description = \(.*?^    return description$",
        lambda _: NEW_DESCRIPTION,
        prompt_section,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not replace the decision prompt safely")
    changed = changed[:prompt_start] + new_section + changed[prompt_end:]

    # Add the signal evaluator only once.
    if "def _deterministic_signal(" not in changed:
        marker = "def _normalize_decisions("
        if marker not in changed:
            raise RuntimeError("Could not find _normalize_decisions function")
        changed = changed.replace(marker, RULE_HELPER + "\n\n" + marker, 1)

    # Add the optional compact snapshot argument.
    sig_old = "    trading_halted: bool = False,\n) -> tuple[list[dict], list[str]]:\n"
    sig_new = "    trading_halted: bool = False,\n    snapshot: dict | None = None,\n) -> tuple[list[dict], list[str]]:\n"
    if sig_new not in changed:
        if sig_old not in changed:
            raise RuntimeError("Could not safely update _normalize_decisions signature")
        changed = changed.replace(sig_old, sig_new, 1)

    # Evaluate deterministic candidates only where the model suggested HOLD.
    injection_marker = "        by_symbol[symbol] = item\n\n    limits = cfg.get(\"risk\", {})\n"
    injected = '''        by_symbol[symbol] = item

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
'''
    if "model HOLD overridden by rule-based" not in changed:
        if injection_marker not in changed:
            raise RuntimeError("Could not find safe insertion point for rule-based candidate checks")
        changed = changed.replace(injection_marker, injected, 1)

    # Pass the actual compact snapshot into normalization.
    call_old = "            raw_decision, symbols, cfg, account, positions, trading_halted\n        )"
    call_new = "            raw_decision, symbols, cfg, account, positions, trading_halted, snapshot=snapshot\n        )"
    if call_new not in changed:
        if call_old not in changed:
            raise RuntimeError("Could not find _normalize_decisions call to pass snapshot")
        changed = changed.replace(call_old, call_new, 1)

    return changed


def main() -> None:
    if not CREW.exists():
        print("Run this script from the project root containing crew\\trading_crew.py", file=sys.stderr)
        sys.exit(2)

    original = CREW.read_text(encoding="utf-8")
    try:
        changed = patch_source(original)
        ast.parse(changed, filename=str(CREW))
    except Exception as exc:
        print(f"Patch aborted; source was not changed: {exc}", file=sys.stderr)
        sys.exit(3)

    backup(CREW)
    CREW.write_text(changed, encoding="utf-8", newline="\n")
    try:
        ast.parse(CREW.read_text(encoding="utf-8"), filename=str(CREW))
    except SyntaxError:
        # Restore automatically if final validation unexpectedly fails.
        bak_files = sorted(CREW.parent.glob("trading_crew.py.*.bak"), key=lambda p: p.stat().st_mtime, reverse=True)
        if bak_files:
            shutil.copy2(bak_files[0], CREW)
        print("Patch failed syntax validation; latest backup restored.", file=sys.stderr)
        sys.exit(4)

    print("Updated crew/trading_crew.py with: corrected MACD semantics, explicit signal thresholds, and per-symbol HOLD diagnostics.")
    print("The update does not edit config, .env, Alpaca paper mode, or risk limits.")
    print("Next: python -m py_compile .\\crew\\trading_crew.py")
    print("Then review git diff and run exactly one paper cycle before any scheduler.")


if __name__ == "__main__":
    main()
