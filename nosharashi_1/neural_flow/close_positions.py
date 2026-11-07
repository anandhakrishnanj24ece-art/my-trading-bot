import json
import os
from datetime import datetime, timezone, timedelta

# This utility force‑closes all open positions without altering the recorded cash balance.
# It reads the adaptive state JSON, zeroes out the weight allocation, records a neutral
# "100% CASH (Neutral)" trade entry and rewrites the state. The balance file is left
# untouched (its previous value is retained).

def close_positions(state_file='paper_state_adaptive.json',
                    balance_file='paper_balance_adaptive.txt'):
    # Load existing state if present
    state = None
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            try:
                state = json.load(f)
            except Exception:
                state = None
    # Determine current portfolio value – default to BASE_CAPITAL if missing
    base_capital = 10000.0
    current_val = state.get('portfolio_value', base_capital) if state else base_capital
    # Zero out all weights (cash position)
    tickers = list(state.get('weights', {}).keys()) if state else []
    zero_weights = {t: 0.0 for t in tickers}
    # Build a neutral trade entry
    now_utc = datetime.now(timezone.utc)
    ist_zone = timezone(timedelta(hours=5, minutes=30))
    now_ist = now_utc.astimezone(ist_zone)
    ist_str = now_ist.strftime('%d %b %Y, %I:%M:%S %p IST')
    trade_entry = {
        'timestamp_ist': ist_str,
        'action': '100% CASH (Neutral)',
        'period_return_pct': 0.0,
        'period_pnl_usd': 0.0,
        'portfolio_value': current_val,
        'total_return_pct': ((current_val - base_capital) / base_capital) * 100.0,
        'leverage': 1.0
    }
    trade_history = state.get('trade_history', []) if state else []
    trade_history.append(trade_entry)
    # Assemble new state – weights set to zero, balance unchanged
    new_state = {
        'portfolio_value': current_val,
        # Preserve the high-water mark; dropping it silently resets the
        # drawdown guard/peak on the next run.
        'peak_equity': float(state.get('peak_equity', current_val)) if state else current_val,
        'weights': zero_weights,
        'weights_raw': zero_weights,
        'last_prices': state.get('last_prices', {}) if state else {},
        'last_updated': now_utc.isoformat(),
        'last_updated_ist': ist_str,
        'last_action': trade_entry['action'],
        'last_period_return_pct': 0.0,
        'last_period_pnl_usd': 0.0,
        'trade_history': trade_history
    }
    if state and 'last_bar_date' in state:
        new_state['last_bar_date'] = state['last_bar_date']
    # Persist state (overwrites previous file)
    with open(state_file, 'w') as f:
        json.dump(new_state, f, indent=2)
    # Do NOT modify the balance file – keep its existing content untouched.
    print(f"[{now_utc.isoformat()}] Closed positions in {state_file}; cash balance unchanged.")

def close_all():
    # Adaptive state
    close_positions('paper_state_adaptive.json', 'paper_balance_adaptive.txt')
    # V3 state (same logic)
    close_positions('paper_state_v3.json', 'paper_balance_v3.txt')

if __name__ == '__main__':
    close_all()
