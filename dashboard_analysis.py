import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from binance.client import Client
from dotenv import load_dotenv

# Load API keys
load_dotenv("key.env")
client = Client(os.getenv("API_KEY"), os.getenv("SECRET_KEY"))

# Use a valid matplotlib style
plt.style.use("seaborn-v0_8-darkgrid")  # Updated seaborn style name

def fetch_dashboard_data():
    positions = client.futures_position_information()
    open_orders = client.futures_get_open_orders()
    open_positions = [p for p in positions if float(p["positionAmt"]) != 0]
    if not open_positions:
        return None

    pos_df = pd.DataFrame(open_positions)
    ord_df = pd.DataFrame(open_orders if open_orders else [])

    for df in [pos_df, ord_df]:
        for col in ['positionAmt', 'entryPrice', 'markPrice', 'unRealizedProfit', 'origQty', 'price', 'stopPrice', 'priceRate', 'activatePrice']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = pd.NA

    data = {
        'symbols': [],
        'unrealized_pnls': [],
        'worst_case_pnls': [],
        'position_values': [],
        'warnings': [],
        'net_unrealized': 0,
        'net_max_loss': 0,
    }

    for sym in pos_df['symbol'].unique():
        p = pos_df[pos_df['symbol'] == sym].iloc[0]
        amt, entry, mark, unreal = p['positionAmt'], p['entryPrice'], p['markPrice'], p['unRealizedProfit']
        side = 'LONG' if amt > 0 else 'SHORT'
        value_usdt = abs(amt * mark)
        data['net_unrealized'] += unreal

        max_loss = 0
        orders = ord_df[ord_df['symbol'] == sym] if not ord_df.empty else pd.DataFrame()
        sl = orders[orders['type'].isin(['STOP_MARKET', 'STOP_LOSS', 'STOP_LOSS_LIMIT'])]
        tsl = orders[orders['type'] == 'TRAILING_STOP_MARKET']
        tp = orders[orders['type'].str.contains('TAKE_PROFIT')]

        covered_qty = sl['origQty'].sum() + tsl['origQty'].sum()
        uncovered = abs(amt) - covered_qty
        if uncovered > 0:
            risk = uncovered * mark
            data['warnings'].append(f"WARNING {sym}: No SL for {uncovered:.2f} units (~${risk:,.2f})")

        for _, s in sl.iterrows():
            if pd.notna(s['stopPrice']):
                loss = abs(entry - s['stopPrice']) * s['origQty']
                if (side == 'LONG' and s['stopPrice'] >= entry) or (side == 'SHORT' and s['stopPrice'] <= entry):
                    loss *= -1
                if abs(loss) > abs(max_loss): max_loss = loss

        for _, t in tsl.iterrows():
            delta, act_price, stop_api = t['priceRate'], t['activatePrice'], t['stopPrice']
            active = pd.isna(act_price) or (mark >= act_price if side == 'LONG' else mark <= act_price)
            if active:
                sl_price = mark * (1 - delta/100) if side == 'LONG' else mark * (1 + delta/100)
            elif pd.notna(stop_api) and stop_api != 0:
                sl_price = stop_api
            else:
                continue
            loss = (entry - sl_price) * t['origQty'] * (1 if side == 'LONG' else -1)
            if abs(loss) > abs(max_loss): max_loss = loss

        if tp.empty:
            data['warnings'].append(f"NOTICE {sym}: No TP orders found.")

        data['symbols'].append(sym)
        data['unrealized_pnls'].append(unreal)
        data['worst_case_pnls'].append(max_loss)
        data['position_values'].append(value_usdt)
        if max_loss < 0:
            data['net_max_loss'] += abs(max_loss)

    return data

fig, axs = plt.subplots(2, 2, figsize=(16, 10))
plt.tight_layout(pad=4.0)
fig.suptitle("Binance Futures Live Dashboard", fontsize=18, fontweight='bold')

def animate(_):
    d = fetch_dashboard_data()
    if not d: return

    axs[0, 0].clear(); axs[0, 1].clear(); axs[1, 0].clear(); axs[1, 1].clear()
    fig.suptitle(f"Binance Futures Dashboard (Updated {time.strftime('%H:%M:%S')})\n"
                 f"Net PnL: ${d['net_unrealized']:+.2f}     Max Loss if SLs Trigger: -${d['net_max_loss']:.2f}",
                 fontsize=16, fontweight='bold')

    # Unrealized PnL bar
    colors = ['green' if x >= 0 else 'red' for x in d['unrealized_pnls']]
    axs[0, 0].bar(d['symbols'], d['unrealized_pnls'], color=colors)
    axs[0, 0].set_title("Unrealized PnL")
    axs[0, 0].set_ylabel("USDT")

    # Worst-case SL loss
    sl_colors = ['red' if x < 0 else 'green' for x in d['worst_case_pnls']]
    axs[0, 1].bar(d['symbols'], d['worst_case_pnls'], color=sl_colors)
    axs[0, 1].set_title("Max SL Loss (Per Coin)")
    axs[0, 1].set_ylabel("USDT")

    # Position value pie chart
    axs[1, 0].pie(d['position_values'], labels=d['symbols'], autopct='%1.1f%%')
    axs[1, 0].set_title("Position Value Allocation")

    # Warning logs
    axs[1, 1].axis('off')
    axs[1, 1].text(0, 1.0, "Warnings", fontsize=12, fontweight='bold', verticalalignment='top')
    if d['warnings']:
        for i, line in enumerate(d['warnings']):
            axs[1, 1].text(0, 0.95 - i*0.08, line, fontsize=10, color='red', verticalalignment='top')
    else:
        axs[1, 1].text(0.5, 0.5, "No SL/TP issues", fontsize=12, color='green', ha='center')

ani = FuncAnimation(fig, animate, interval=2000)
plt.show()
