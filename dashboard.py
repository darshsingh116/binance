import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from binance.client import Client
import streamlit as st
from streamlit_js_eval import streamlit_js_eval
import base64

def encode_secret(secret):
    return base64.b64encode(secret.encode()).decode()

def decode_secret(encoded):
    try:
        return base64.b64decode(encoded.encode()).decode()
    except Exception:
        return ""

def check_or_ask_keys():
    if "API_KEY" in st.session_state and "SECRET_KEY" in st.session_state:
        return True

    # Try reading from browser localStorage
    stored = streamlit_js_eval(js_expressions="localStorage.getItem('binance_keys')", key="get_keys")
    if stored:
        try:
            key, secret = stored.split("::")
            st.session_state["API_KEY"] = decode_secret(key)
            st.session_state["SECRET_KEY"] = decode_secret(secret)
            return True
        except:
            pass

    # Ask for input if not found
    with st.form("🔐 Enter Binance API Keys"):
        st.subheader("🔑 API Authentication")
        api = st.text_input("Binance API Key", type="password")
        secret = st.text_input("Binance Secret Key", type="password")
        save = st.checkbox("Remember on this browser (store in localStorage)", value=True)
        submit = st.form_submit_button("Submit")

        if submit:
            if api and secret:
                st.session_state["API_KEY"] = api
                st.session_state["SECRET_KEY"] = secret
                if save:
                    # Store in localStorage encoded
                    encoded = f"{encode_secret(api)}::{encode_secret(secret)}"
                    streamlit_js_eval(js_expressions=f"localStorage.setItem('binance_keys', '{encoded}')", key="set_keys")
                st.success("✅ API Keys saved. Loading dashboard...")
                st.rerun()
                st.error("❌ Please enter both API key and Secret.")

    return False


def logout():
    st.session_state.pop("API_KEY", None)
    st.session_state.pop("SECRET_KEY", None)
    # Clear from browser
    streamlit_js_eval(js_expressions="localStorage.removeItem('binance_keys')", key="clear_keys")
    st.success("Logged out successfully.")
    st.rerun()
# # --- Load API Keys ---
# load_dotenv("key.env")
# API_KEY = os.getenv("API_KEY")
# SECRET_KEY = os.getenv("SECRET_KEY")
# client = Client(API_KEY, SECRET_KEY)

# from dotenv import load_dotenv
# load_dotenv("key.env")
# API_KEY = os.getenv("API_KEY")
# SECRET_KEY = os.getenv("SECRET_KEY")

# --- Use dynamic key management instead ---
# ---- Key utils ----

# 🔐 Now check for API keys (ask if not found)
if not check_or_ask_keys():
    st.stop()

# ✅ Place this block AFTER authentication check:
with st.sidebar:
    st.markdown("### 👤 Session")
    if "API_KEY" in st.session_state:
        st.write("🔐 Logged in")
        if st.button("🚪 Logout"):
            logout()

API_KEY = st.session_state["API_KEY"]
SECRET_KEY = st.session_state["SECRET_KEY"]
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
    fig, ax = plt.subplots(figsize=(4, 3))
    if not sl_orders:
        return fig
    
    # Filter out regular stop losses if there's an active trailing stop
    has_active_trailing = any('trail_percent' in o and 'ACTIVE' in o['type'] for o in sl_orders)
    filtered_orders = []
    
    for o in sl_orders:
        if 'trail_percent' in o:
            # Always show trailing stops
            filtered_orders.append(o)
        else:
            # Only show regular stops if no active trailing stop
            if not has_active_trailing:
                filtered_orders.append(o)
            # If there's an active trailing stop, mark regular stop as conflicting
            else:
                o['conflicting'] = True
                filtered_orders.append(o)
    
    labels = []
    values = []  # Can be losses (positive) or profits (negative)
    colors = []
    
    for o in filtered_orders:
        if o.get('conflicting'):
            # Mark conflicting orders with cross-off
            labels.append(f"❌ {o['type'][:4]}\n@ {o['stopPrice']:.4f}")
            colors.append('lightgray')
            values.append(0)  # Don't show value for conflicting
        elif 'trail_percent' in o:
            # Trailing stop
            if 'ACTIVE' in o['type']:
                if o['profit'] > 0:  # Profit
                    labels.append(f"TSL {o['trail_percent']:.1f}%\n@ {o['stopPrice']:.4f}")
                    colors.append('grey')
                    values.append(-o['profit'])  # Negative for profit display
                elif o['loss'] == 0:  # Break-even
                    labels.append(f"TSL {o['trail_percent']:.1f}%\n@ {o['stopPrice']:.4f}")
                    colors.append('grey')
                    values.append(0)
                else:
                    labels.append(f"TSL {o['trail_percent']:.1f}%\n@ {o['stopPrice']:.4f}")
                    colors.append('orange')
                    values.append(o['loss'])
            else:
                labels.append(f"TSL {o['trail_percent']:.1f}%\n(Inactive)")
                colors.append('lightcoral')
                values.append(o['loss'])
        else:
            # Regular stop (only if no conflicting)
            labels.append(f"{o['type'][:4]}\n@ {o['stopPrice']:.4f}")
            colors.append('red')
            values.append(o['loss'])
    
    bars = ax.bar(labels, values, color=colors)
    
    for bar, val, order in zip(bars, values, filtered_orders):
        if order.get('conflicting'):
            # Position CANCEL text to the right of the bar instead of above
            ax.text(bar.get_x() + bar.get_width() + 0.05, 0, 
                    "CANCEL", ha='left', va='center', fontsize=7, fontweight='bold', color='red')
        elif val > 0:
            # Loss - position to the right of bar
            ax.text(bar.get_x() + bar.get_width() + 0.05, bar.get_height() / 2, 
                    f"-${val:.2f}", ha='left', va='center', fontsize=8, fontweight='bold')
        elif val == 0:
            # Break-even - position to the right of bar
            ax.text(bar.get_x() + bar.get_width() + 0.05, 0, 
                    "$0.00", ha='left', va='center', fontsize=7, fontweight='bold', color='grey')
        else:
            # Profit - position to the right of bar
            ax.text(bar.get_x() + bar.get_width() + 0.05, bar.get_height() / 2, 
                    f"+${abs(val):.2f}", ha='left', va='center', fontsize=8, fontweight='bold', color='green')
    
    ax.set_ylabel("Loss/Profit (USDT)")
    ax.set_title(f"{symbol} SL Risk Analysis", fontsize=10, fontweight='bold')
    
    # Set y-axis limits to reduce white space
    if values:
        y_min = min(values) if min(values) < 0 else 0
        y_max = max(values) if max(values) > 0 else 0
        # Add small padding
        padding = max(abs(y_min), abs(y_max)) * 0.1 if (y_min != 0 or y_max != 0) else 10
        ax.set_ylim(y_min - padding, y_max + padding)
    
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    return fig


def draw_total_pnl_chart(per_coin_data):
    symbols = [coin['symbol'] for coin in per_coin_data]
    pnls = [coin['unrealized_pnl'] for coin in per_coin_data]
    
    # Calculate dynamic max losses (considering which SL will trigger first)
    max_losses = []
    profit_values = []  # Track profit values separately
    
    for coin in per_coin_data:
        if coin['sl_orders']:
            # Find the SL that will trigger first (closest to current price)
            current_price = coin['mark_price']
            side = coin['position_side']
            
            closest_sl = None
            closest_distance = float('inf')
            
            # Filter out conflicting orders
            has_active_trailing = any('trail_percent' in sl and 'ACTIVE' in sl['type'] for sl in coin['sl_orders'])
            valid_orders = []
            for sl in coin['sl_orders']:
                if 'trail_percent' in sl:
                    valid_orders.append(sl)  # Always include trailing stops
                elif not has_active_trailing:
                    valid_orders.append(sl)  # Only include regular stops if no active trailing
            
            for sl in valid_orders:
                # Calculate distance considering direction
                if side == 'LONG':
                    # For LONG positions, SL triggers when price goes down
                    if sl['stopPrice'] < current_price:
                        distance = current_price - sl['stopPrice']
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_sl = sl
                else:
                    # For SHORT positions, SL triggers when price goes up
                    if sl['stopPrice'] > current_price:
                        distance = sl['stopPrice'] - current_price
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_sl = sl
            
            # Use the loss/profit from the closest SL
            if closest_sl:
                if closest_sl.get('profit', 0) > 0:
                    max_losses.append(0)  # No loss if profitable
                    profit_values.append(closest_sl['profit'])
                else:
                    max_loss = closest_sl['loss']
                    max_losses.append(-max_loss)  # Negative for below x-axis
                    profit_values.append(0)
            else:
                # Fallback to worst case loss
                max_loss = coin['max_loss'] if coin['max_loss'] > 0 else 0
                max_losses.append(-max_loss)
                profit_values.append(0)
        else:
            max_losses.append(0)
            profit_values.append(0)

    fig, ax = plt.subplots(figsize=(6, 2.5))
    x = range(len(symbols))
    
    # Plot max losses (red/grey bars below zero)
    loss_colors = ['lightgrey' if loss < 0 else 'white' for loss in max_losses]
    ax.bar(x, max_losses, width=0.4, color=loss_colors, label='Max SL Loss (Next to Trigger)', alpha=0.7)
    
    # Plot profit values (green bars above zero)
    profit_bars = ax.bar(x, profit_values, width=0.3, color='lightgreen', label='SL Profit Potential', alpha=0.8)
    
    # Plot current unrealized PnL
    pnl_bars = ax.bar(x, pnls, width=0.2, color='green', label='Current Unrealized PnL')

    # Add value labels positioned beside bars to avoid overlap
    for i, pnl in enumerate(pnls):
        # Position PnL labels to the right of bars
        ax.text(x[i] + 0.1, pnl, f"${pnl:.2f}", ha='left', va='center', fontsize=7, fontweight='bold', color='darkgreen')
    
    for i, loss in enumerate(max_losses):
        if loss != 0:
            # Position loss labels to the left of bars
            ax.text(x[i] - 0.1, loss, f"-${abs(loss):.2f}", ha='right', va='center', fontsize=7, color='grey')
    
    for i, profit in enumerate(profit_values):
        if profit > 0:
            # Position profit labels to the right of bars
            ax.text(x[i] + 0.1, profit, f"+${profit:.2f}", ha='left', va='center', fontsize=7, color='green', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(symbols, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel("PnL (USDT)")
    ax.set_title("Unrealized PnL vs Next SL to Trigger", fontsize=9)
    ax.legend(fontsize=7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)  # Zero line
    
    # Set y-axis limits to reduce white space
    all_values = pnls + max_losses + profit_values
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        # Add small padding (10% of the range)
        y_range = y_max - y_min
        padding = y_range * 0.1 if y_range > 0 else max(abs(y_min), abs(y_max)) * 0.1
        if padding == 0:
            padding = 10  # Minimum padding for very small values
        ax.set_ylim(y_min - padding, y_max + padding)
    
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
    processed_symbols = set()  # Track processed symbols to avoid duplicates

    for symbol in positions_df['symbol'].unique():
        if symbol in processed_symbols:
            continue  # Skip if already processed
        processed_symbols.add(symbol)
        
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

        # Process regular stop loss orders
        for _, sl in sl_orders.iterrows():
            if pd.notna(sl['stopPrice']):
                # Calculate loss correctly for regular stops
                if side == 'LONG':
                    loss = (entry - sl['stopPrice']) * sl['origQty']
                    loss_value = max(0, loss)  # Ensure positive loss
                else:
                    loss = (sl['stopPrice'] - entry) * sl['origQty']
                    loss_value = max(0, loss)  # Ensure positive loss
                
                sl_orders_list.append({
                    'type': sl['type'],
                    'stopPrice': sl['stopPrice'],
                    'loss': loss_value,
                    'profit': 0,
                    'qty': sl['origQty']
                })
                if loss_value > coin_max_loss:
                    coin_max_loss = loss_value

        # Process trailing stop loss orders
        trailing_sl_orders = orders[orders['type'] == 'TRAILING_STOP_MARKET']
        for _, tsl in trailing_sl_orders.iterrows():
            if pd.notna(tsl['priceRate']) and pd.notna(tsl['origQty']):
                delta_percent = tsl['priceRate']
                activate_price = tsl['activatePrice']
                stop_price_from_api = tsl['stopPrice']
                
                # Determine if trailing stop is active
                if side == 'LONG':
                    activation_condition_met = pd.isna(activate_price) or mark >= activate_price
                else:
                    activation_condition_met = pd.isna(activate_price) or mark <= activate_price
                
                if activation_condition_met:
                    # Trailing stop is active - calculate current trailing price
                    if side == 'LONG':
                        effective_sl_price = mark * (1 - delta_percent / 100)
                    else:
                        effective_sl_price = mark * (1 + delta_percent / 100)
                    
                    # Calculate loss/profit correctly for trailing stops
                    # For trailing stops, calculate P&L from entry to current trailing price
                    if side == 'LONG':
                        # LONG: calculate P&L when price drops to trailing SL
                        pnl_from_entry = (effective_sl_price - entry) * tsl['origQty']
                    else:
                        # SHORT: calculate P&L when price rises to trailing SL  
                        pnl_from_entry = (entry - effective_sl_price) * tsl['origQty']
                    
                    # Determine if it's profit, break-even, or loss
                    if pnl_from_entry > 0:
                        # Profit scenario
                        sl_orders_list.append({
                            'type': 'TRAILING_STOP_MARKET (ACTIVE)',
                            'stopPrice': effective_sl_price,
                            'loss': 0,
                            'profit': pnl_from_entry,
                            'qty': tsl['origQty'],
                            'trail_percent': delta_percent
                        })
                    elif pnl_from_entry == 0:
                        # Break-even scenario
                        sl_orders_list.append({
                            'type': 'TRAILING_STOP_MARKET (ACTIVE)',
                            'stopPrice': effective_sl_price,
                            'loss': 0,
                            'profit': 0,
                            'qty': tsl['origQty'],
                            'trail_percent': delta_percent
                        })
                    else:
                        # Loss scenario
                        loss_from_entry = abs(pnl_from_entry)
                        sl_orders_list.append({
                            'type': 'TRAILING_STOP_MARKET (ACTIVE)',
                            'stopPrice': effective_sl_price,
                            'loss': loss_from_entry,
                            'profit': 0,
                            'qty': tsl['origQty'],
                            'trail_percent': delta_percent
                        })
                        
                        if loss_from_entry > coin_max_loss:
                            coin_max_loss = loss_from_entry
                        
                else:
                    # Trailing stop not yet active
                    if pd.notna(stop_price_from_api) and stop_price_from_api != 0:
                        # Use the current stop price from API
                        if side == 'LONG':
                            loss_from_entry = (entry - stop_price_from_api) * tsl['origQty']
                            loss_from_entry = max(0, loss_from_entry)
                        else:
                            loss_from_entry = (stop_price_from_api - entry) * tsl['origQty']
                            loss_from_entry = max(0, loss_from_entry)
                        
                        sl_orders_list.append({
                            'type': 'TRAILING_STOP_MARKET (INACTIVE)',
                            'stopPrice': stop_price_from_api,
                            'loss': loss_from_entry,
                            'profit': 0,
                            'qty': tsl['origQty'],
                            'trail_percent': delta_percent,
                            'activate_at': activate_price
                        })
                        
                        if loss_from_entry > coin_max_loss:
                            coin_max_loss = loss_from_entry

        # Calculate which SL will actually trigger first (for net loss calculation)
        actual_trigger_loss = 0
        actual_trigger_profit = 0
        
        if sl_orders_list:
            current_price = mark
            closest_sl = None
            closest_distance = float('inf')
            
            # Filter out conflicting orders for net calculation
            has_active_trailing = any('trail_percent' in sl and 'ACTIVE' in sl['type'] for sl in sl_orders_list)
            valid_orders = []
            conflicting_orders = []
            
            for sl in sl_orders_list:
                if 'trail_percent' in sl:
                    valid_orders.append(sl)  # Always include trailing stops
                elif not has_active_trailing:
                    valid_orders.append(sl)  # Only include regular stops if no active trailing
                else:
                    conflicting_orders.append(sl)  # Mark regular stops as conflicting
            
            for sl in valid_orders:
                # Calculate distance considering direction
                if side == 'LONG':
                    # For LONG positions, SL triggers when price goes down
                    if sl['stopPrice'] < current_price:
                        distance = current_price - sl['stopPrice']
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_sl = sl
                else:
                    # For SHORT positions, SL triggers when price goes up
                    if sl['stopPrice'] > current_price:
                        distance = sl['stopPrice'] - current_price
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_sl = sl
            
            # Use the loss/profit from the closest SL that can actually trigger
            if closest_sl:
                actual_trigger_loss = closest_sl.get('loss', 0)
                actual_trigger_profit = closest_sl.get('profit', 0)
            else:
                # Fallback to worst case if no directional SL found
                actual_trigger_loss = coin_max_loss

        # Add the actual trigger loss to net total
        if actual_trigger_loss > 0:
            net_max_loss_from_stops += actual_trigger_loss

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

# Direct rendering without placeholder
st.subheader("Live Overview (auto-refresh every 2 sec)")
coins, net_pnl, net_loss, net_tp = analyze_futures_account()

if coins:  # Only show if we have data
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
            
            # Check for conflicting orders
            has_active_trailing = any('trail_percent' in sl and 'ACTIVE' in sl['type'] for sl in coin['sl_orders'])
            conflicting_orders = []
            
            if has_active_trailing:
                conflicting_orders = [sl for sl in coin['sl_orders'] if 'trail_percent' not in sl]
                if conflicting_orders:
                    st.warning(f"⚠️ **CONFLICTING ORDERS DETECTED:** You have {len(conflicting_orders)} regular stop loss order(s) that conflict with your active trailing stop. Consider canceling these regular stop losses:")
                    for sl in conflicting_orders:
                        st.markdown(f"   - ❌ {sl['type']} @ ${sl['stopPrice']:.4f} → **CANCEL THIS ORDER**")
            
            if coin['sl_orders']:
                st.markdown("**🛑 STOP LOSS ORDERS:**")
                for sl in coin['sl_orders']:
                    # Skip conflicting orders in main display (they're shown in warning above)
                    if sl in conflicting_orders:
                        continue
                        
                    # Handle different display for losses vs profits/break-even
                    if 'trail_percent' in sl:
                        # Trailing stop loss
                        if 'ACTIVE' in sl['type']:
                            if sl.get('profit', 0) > 0:
                                st.markdown(f"- {sl['type']} @ ${sl['stopPrice']:.4f} (Trail: {sl['trail_percent']:.2f}%) → **+${sl['profit']:.2f} Profit** 🎯")
                            elif sl['loss'] == 0:
                                st.markdown(f"- {sl['type']} @ ${sl['stopPrice']:.4f} (Trail: {sl['trail_percent']:.2f}%) → **$0.00 Break-even**")
                            else:
                                st.markdown(f"- {sl['type']} @ ${sl['stopPrice']:.4f} (Trail: {sl['trail_percent']:.2f}%) → -${sl['loss']:.2f}")
                        else:
                            st.markdown(f"- {sl['type']} @ ${sl['stopPrice']:.4f} (Trail: {sl['trail_percent']:.2f}%, Activates @ ${sl.get('activate_at', 'N/A')}) → -${sl['loss']:.2f}")
                    else:
                        # Regular stop loss
                        st.markdown(f"- {sl['type']} @ ${sl['stopPrice']:.4f} → -${sl['loss']:.2f}")
                
                # Show which SL will trigger first (excluding conflicting orders)
                valid_sl_orders = [sl for sl in coin['sl_orders'] if sl not in conflicting_orders]
                if len(valid_sl_orders) > 1:
                    closest_sl = None
                    closest_distance = float('inf')
                    current_price = coin['mark_price']
                    
                    for sl in valid_sl_orders:
                        distance = abs(current_price - sl['stopPrice'])
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_sl = sl
                    
                    if closest_sl:
                        st.info(f"⚡ **Closest to trigger:** {closest_sl['type']} @ ${closest_sl['stopPrice']:.4f} (Distance: ${closest_distance:.4f})")
                
                st.pyplot(draw_loss_chart(coin['sl_orders'], coin['symbol']))
            else:
                # Calculate uncovered quantity
                total_sl_qty = sum(sl.get('qty', 0) for sl in coin['sl_orders'])
                uncovered_qty = abs(coin['position_amt']) - total_sl_qty
                
                if uncovered_qty > 0:
                    uncovered_value = uncovered_qty * coin['mark_price']
                    st.warning(f"⚠️ No SL for {uncovered_qty:.4f} units (~${uncovered_value:,.0f}). Risk is undefined.")
                else:
                    st.warning(f"⚠️ No stop loss orders found for {abs(coin['position_amt']):.4f} units. Risk is undefined.")

# Auto-refresh with delay
time.sleep(2)
st.rerun()
