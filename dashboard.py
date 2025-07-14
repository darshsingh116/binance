import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from binance.client import Client

# --- Load API Keys ---
load_dotenv("key.env")
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
client = Client(API_KEY, SECRET_KEY)


def fetch_account_data():
    positions = client.futures_position_information()
    open_orders = client.futures_get_open_orders()

    open_positions = [p for p in positions if float(p['positionAmt']) != 0]
    if not open_positions:
        return None, None

    positions_df = pd.DataFrame(open_positions)
    orders_df = pd.DataFrame(open_orders) if open_orders else pd.DataFrame()

    for df in [positions_df, orders_df]:
        if df.empty:
            continue
        for col in ['positionAmt', 'entryPrice', 'markPrice', 'unRealizedProfit',
                    'origQty', 'price', 'stopPrice', 'priceRate', 'activatePrice']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = pd.NA

    return positions_df, orders_df


def draw_loss_chart(sl_orders, symbol):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    if not sl_orders:
        return fig
    labels = [f"{o['type']} @ {o['stopPrice']}" for o in sl_orders]
    losses = [abs(o['loss']) for o in sl_orders]
    bars = ax.bar(labels, losses, color='red')
    for bar, val in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"-${val:.2f}", ha='center', va='bottom', fontsize=7)
    ax.set_ylabel("Loss (USDT)")
    ax.set_title(f"{symbol} SL Risk", fontsize=9)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.tight_layout()
    return fig


def draw_total_pnl_chart(per_coin_data):
    symbols = [coin['symbol'] for coin in per_coin_data]
    pnls = [coin['unrealized_pnl'] for coin in per_coin_data]
    max_losses = [-abs(coin['max_loss']) for coin in per_coin_data]  # Negative for below x-axis

    fig, ax = plt.subplots(figsize=(6, 2.5))
    x = range(len(symbols))
    ax.bar(x, max_losses, width=0.4, color='lightgrey', label='Max SL Loss')  # Now negative
    bars = ax.bar(x, pnls, width=0.3, color='green', label='Unrealized PnL')

    for i, (bar, pnl) in enumerate(zip(bars, pnls)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"${pnl:.2f}", ha='center', va='bottom', fontsize=7)
    for i, loss in enumerate(max_losses):
        if loss != 0:
            ax.text(x[i], loss, f"-${abs(loss):.2f}", ha='center', va='top', fontsize=7, color='grey')

    ax.set_xticks(x)
    ax.set_xticklabels(symbols, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel("PnL (USDT)")
    ax.set_title("Unrealized PnL vs SL Risk", fontsize=9)
    ax.legend(fontsize=7)
    plt.tight_layout()
    return fig


def analyze_futures_account():
    positions_df, orders_df = fetch_account_data()
    if positions_df is None:
        st.warning("No open futures positions.")
        return [], 0, 0, 0

    total_potential_profit = 0
    total_current_unrealized_pnl = 0
    net_max_loss_from_stops = 0
    per_coin_data = []

    for symbol in positions_df['symbol'].unique():
        position = positions_df[positions_df['symbol'] == symbol].iloc[0]
        amt = position['positionAmt']
        entry = position['entryPrice']
        mark = position['markPrice']
        side = 'LONG' if amt > 0 else 'SHORT'
        value = abs(amt * mark)
        unreal = position['unRealizedProfit']
        total_current_unrealized_pnl += unreal

        orders = orders_df[orders_df['symbol'] == symbol] if not orders_df.empty else pd.DataFrame()
        coin_max_loss = 0
        sl_orders_list = []

        sl_orders = orders[
            orders['type'].isin(['STOP_MARKET', 'STOP_LOSS', 'STOP_LOSS_LIMIT']) &
            (orders['type'] != 'TRAILING_STOP_MARKET')
        ]

        for _, sl in sl_orders.iterrows():
            if pd.notna(sl['stopPrice']):
                loss = abs(entry - sl['stopPrice']) * sl['origQty']
                if side == 'LONG' and sl['stopPrice'] >= entry:
                    loss_value = -loss
                elif side == 'SHORT' and sl['stopPrice'] <= entry:
                    loss_value = -loss
                else:
                    loss_value = loss
                sl_orders_list.append({
                    'type': sl['type'],
                    'stopPrice': sl['stopPrice'],
                    'loss': loss_value
                })
                if abs(loss_value) > abs(coin_max_loss):
                    coin_max_loss = loss_value

        net_max_loss_from_stops += max(0, coin_max_loss)

        tp_orders = orders[orders['type'].isin(['TAKE_PROFIT', 'TAKE_PROFIT_LIMIT', 'TAKE_PROFIT_MARKET'])]
        max_profit = 0
        for _, tp in tp_orders.iterrows():
            if pd.notna(tp['stopPrice']):
                profit = abs(tp['stopPrice'] - entry) * tp['origQty']
                max_profit += profit
        total_potential_profit += max_profit

        per_coin_data.append({
            'symbol': symbol,
            'unrealized_pnl': unreal,
            'max_loss': coin_max_loss,
            'sl_orders': sl_orders_list,
            'position_amt': amt,
            'entry_price': entry,
            'mark_price': mark,
            'position_side': side,
            'position_value_usdt': value,
        })

    return per_coin_data, total_current_unrealized_pnl, net_max_loss_from_stops, total_potential_profit


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📈 Binance Futures Position Dashboard")
placeholder = st.empty()

while True:
    with placeholder.container():
        st.subheader("Live Overview (auto-refresh every 2 sec)")
        coins, net_pnl, net_loss, net_tp = analyze_futures_account()

        col1, col2, col3 = st.columns(3)
        col1.metric("Net Unrealized PnL", f"${net_pnl:,.2f}")
        col2.metric("Net Max SL Loss", f"-${net_loss:,.2f}")
        col3.metric("Total TP Profit", f"+${net_tp:,.2f}")

        st.pyplot(draw_total_pnl_chart(coins))

        for coin in coins:
            with st.expander(f"🔎 {coin['symbol']} Details"):
                st.markdown(f"""
                **✅ OPEN POSITION:**
                - Side: {coin['position_side']}, Quantity: {coin['position_amt']}, Value: ${coin['position_value_usdt']:,.2f}
                - Entry Price: ${coin['entry_price']:.4f}, Mark Price: ${coin['mark_price']:.4f}
                - Unrealized PnL: {'+' if coin['unrealized_pnl'] >= 0 else '-'}${abs(coin['unrealized_pnl']):,.2f}
                """)
                if coin['sl_orders']:
                    st.markdown("**🛑 FIXED STOP LOSS ORDERS:**")
                    for sl in coin['sl_orders']:
                        sign = '+' if sl['loss'] < 0 else '-'
                        st.markdown(f"- {sl['type']} @ ${sl['stopPrice']:.4f} → {sign}${abs(sl['loss']):.2f}")
                    st.pyplot(draw_loss_chart(coin['sl_orders'], coin['symbol']))
                else:
                    st.warning(f"⚠️ No SL found for {abs(coin['position_amt'])} units. Risk is undefined.")

    time.sleep(2)
    st.rerun()
