# Compact Groq trading patch

Apply this archive at the root of the existing trading-agent-group repository.
It replaces only the paths included here. It intentionally contains no .env,
API keys, Git history, logs, virtual environment, or caches.

Key changes:
- One Groq model inference for a compact portfolio decision JSON.
- Fetch OHLCV, portfolio snapshot, current paper account, current holdings, and safety state via direct MCP calls (no LLM tool calls).
- New calculate_portfolio_snapshot tool calculates each stock's indicators and beta locally and returns compact rows.
- Python validates JSON and limits buys to position headroom and cash buffer; it won't sell more than an existing long position.
- Malformed model output fails closed to HOLD; active/unknown safety state blocks buys.
- Example and Aman configs use groq/openai/gpt-oss-20b, max_tokens=1200, 120 calendar days/60-day beta window, and hedging disabled.
- Server-side Alpaca paper-order validation and daily-loss halt remain enabled.
- Regression/IV and option-chain tools remain available in the MCP server but are removed from the default decision path.

Validation: Python AST/compile checks and YAML parsing passed. Unit smoke checks covered buy caps, held-position sell validation, halt behavior, and malformed-JSON fallback. The project was not run end-to-end against Groq or Alpaca from this environment.
