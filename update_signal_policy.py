from __future__ import annotations

import ast
import re
import shutil
from datetime import datetime
from pathlib import Path

TARGET = Path('crew/trading_crew.py')

NEW_HELPER = r'''def _deterministic_signal(
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

    if 40.0 <= rsi <= 65.0:
        buy_score += 1.0
    elif 35.0 <= rsi < 40.0 or 65.0 < rsi <= 68.0:
        buy_score += 0.5
    elif rsi >= 70.0 or (0.0 < rsi < 30.0):
        buy_score -= 2.0

    if 0.2 <= bb <= 0.8:
        buy_score += 1.0
    elif 0.1 <= bb < 0.2 or 0.8 < bb <= 0.9:
        buy_score += 0.25
    elif bb < 0.0 or bb > 1.0:
        buy_score -= 1.0

    buy_rsi_ok = 35.0 <= rsi <= 68.0
    # Confirmation is an actual bullish crossover OR constructive position
    # inside the Bollinger range. No full-day volume comparison intraday.
    buy_confirmation = (macd == "bullish") or (0.15 <= bb <= 0.85)
    buy_eligible = (
        not trading_halted
        and trend == "up"
        and macd != "bearish"
        and buy_rsi_ok
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
        f"MACD={row.get('macd', 'missing')}, %B={round(bb, 3) if -10 < bb < 10 else 'missing'}, "
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
        why.append(f"RSI={rsi if rsi >= 0 else 'missing'} outside 35-68")
    if macd == "bearish":
        why.append("MACD crossover bearish")
    if not buy_confirmation:
        why.append("no bullish MACD crossover or constructive Bollinger position")
    if buy_score < 3.0:
        why.append(f"buy score {buy_score:.1f} below 3.0")
    if held_value < 1.0:
        why.append("no held position to exit")
    why.append(diagnostics)
    return {
        "action": "hold", "buy_score": round(buy_score, 1),
        "sell_score": round(sell_score, 1), "reason": "; ".join(why),
    }
'''

NEW_DESCRIPTION = r'''    description = (
        "Evaluate each symbol individually for an Alpaca PAPER account using only the supplied "
        "market snapshot, positions, account, and limits. Trend is 'Up', 'Down', or 'Flat'. "
        "The macd field is a crossover category: 'Bullish', 'Bearish', or 'None'; 'None' means "
        "no fresh crossover, not bearish. RSI is numeric and bb_percent_b is a Bollinger position "
        "value normally between 0 and 1. relative_volume is diagnostic only: because the compact "
        "workflow uses 1Day bars, the current bar's volume may be partial during market hours and "
        "must not be treated as a full-session relative-volume reading.\n\n"
        "ENTRY RULE: consider BUY if trend is 'Up', RSI is 35-68, MACD is not 'Bearish', "
        "and either MACD='Bullish' or bb_percent_b is between 0.15 and 0.85. Favor HOLD if the "
        "indicators conflict, the trend is not Up, RSI is outside the range, or required data is "
        "missing. A fresh MACD crossover is not required when the Bollinger position is constructive. "
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
    return description'''


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"Run this script from the project root; not found: {TARGET}")
    source = TARGET.read_text(encoding="utf-8")
    prompt_start = source.find("def _decision_task_description(")
    prompt_end = source.find("\ndef run_one_cycle(", prompt_start)
    if prompt_start < 0 or prompt_end < 0:
        raise SystemExit("Could not locate the decision prompt; no files changed.")
    prompt_section = source[prompt_start:prompt_end]
    prompt_section_new, prompt_count = re.subn(
        r"(?ms)^    description = \(.*?^    return description$",
        lambda _: NEW_DESCRIPTION,
        prompt_section,
        count=1,
    )
    if prompt_count != 1:
        raise SystemExit("Could not safely update the decision prompt; no files changed.")
    source = source[:prompt_start] + prompt_section_new + source[prompt_end:]

    start = source.find("def _deterministic_signal(")
    end = source.find("\ndef _normalize_decisions(", start)
    if start < 0 or end < 0:
        raise SystemExit("Could not locate the existing deterministic signal helper; no files changed.")
    revised = source[:start] + NEW_HELPER.rstrip() + source[end:]
    compile(revised, str(TARGET), "exec")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = TARGET.with_suffix(TARGET.suffix + f".{stamp}.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(revised, encoding="utf-8", newline="\n")
    # Validate the saved content and confirm the only target file is the crew file.
    ast.parse(TARGET.read_text(encoding="utf-8"))
    print(f"Backup created: {backup}")
    print(f"Updated: {TARGET}")
    print("The rule gate now ignores partial daily volume as a hard signal and reports raw indicator values.")
    print("Existing risk limits and Alpaca paper-only configuration were not modified.")

if __name__ == "__main__":
    main()
