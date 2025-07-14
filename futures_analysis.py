import pandas as pd
from binance.client import Client
import os
from dotenv import load_dotenv

# --- Load API Keys from .env file ---
load_dotenv("key.env")
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

# --- Initialize Binance Client ---
client = Client(API_KEY, SECRET_KEY)

def analyze_futures_account():
    try:
        positions = client.futures_position_information()
        open_orders = client.futures_get_open_orders()

        open_positions = [p for p in positions if float(p['positionAmt']) != 0]
        if not open_positions:
            print("No open futures positions found.")
            return

        positions_df = pd.DataFrame(open_positions)
        orders_df = pd.DataFrame(open_orders) if open_orders else pd.DataFrame()

        for df in [positions_df, orders_df]:
            if df.empty: continue
            for col in ['positionAmt', 'entryPrice', 'markPrice', 'unRealizedProfit', 'origQty', 'price', 'stopPrice', 'priceRate', 'activatePrice']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    df[col] = pd.NA

        total_potential_profit = 0
        total_current_unrealized_pnl = 0
        net_max_loss_from_stops = 0

        unique_symbols = positions_df['symbol'].unique()
        print("--- Account Risk & Profit Analysis ---\n")

        for symbol in unique_symbols:
            print(f"==================== {symbol} ====================")

            position = positions_df[positions_df['symbol'] == symbol].iloc[0]
            position_amt = position['positionAmt']
            entry_price = position['entryPrice']
            mark_price = position['markPrice']
            position_side = 'LONG' if position_amt > 0 else 'SHORT'
            position_value_usdt = abs(position_amt * mark_price)
            unrealized_profit = position['unRealizedProfit']
            total_current_unrealized_pnl += unrealized_profit

            pnl_sign = '+' if unrealized_profit >= 0 else '-'
            print(f"✅ OPEN POSITION:")
            print(f"  - Side: {position_side}, Quantity: {position_amt}, Value: ${position_value_usdt:,.2f}")
            print(f"  - Entry Price: ${entry_price:,.4f}, Mark Price: ${mark_price:,.4f}")
            print(f"  - Current Unrealized PnL: {pnl_sign}${abs(unrealized_profit):,.2f}\n")

            associated_orders = orders_df[orders_df['symbol'] == symbol] if not orders_df.empty else pd.DataFrame()
            coin_max_loss = 0
            covered_qty_by_fixed_sl = 0

            sl_orders = associated_orders[
                associated_orders['type'].isin(['STOP_MARKET', 'STOP_LOSS', 'STOP_LOSS_LIMIT']) &
                (associated_orders['type'] != 'TRAILING_STOP_MARKET')
            ]

            if not sl_orders.empty:
                print(f"🛑 FIXED STOP LOSS ORDERS:")
                for _, sl in sl_orders.iterrows():
                    if pd.notna(sl['stopPrice']):
                        loss_from_entry = abs(entry_price - sl['stopPrice']) * sl['origQty']
                        if position_side == 'LONG' and sl['stopPrice'] >= entry_price:
                            loss_value = -loss_from_entry
                        elif position_side == 'SHORT' and sl['stopPrice'] <= entry_price:
                            loss_value = -loss_from_entry
                        else:
                            loss_value = loss_from_entry

                        sign = '+' if loss_value < 0 else '-'
                        print(f"  - Type: {sl['type']}, Quantity: {sl['origQty']}, Trigger: ${sl['stopPrice']:,.4f} -> {sign}${abs(loss_value):,.2f}")

                        if abs(loss_value) > abs(coin_max_loss):
                            coin_max_loss = loss_value
                        covered_qty_by_fixed_sl += sl['origQty']

            trailing_sl_orders = associated_orders[associated_orders['type'] == 'TRAILING_STOP_MARKET']

            if not trailing_sl_orders.empty:
                print(f"🟠 TRAILING STOP LOSS ORDERS:")
                for _, tsl in trailing_sl_orders.iterrows():
                    delta_percent = tsl['priceRate']
                    activate_price = tsl['activatePrice']
                    stop_price_from_api = tsl['stopPrice']
                    effective_sl_price = None
                    loss_from_entry_at_sl = None

                    if position_side == 'LONG':
                        activation_condition_met = pd.isna(activate_price) or mark_price >= activate_price
                    else:
                        activation_condition_met = pd.isna(activate_price) or mark_price <= activate_price

                    if not activation_condition_met:
                        print(f"  - Type: {tsl['type']}, Quantity: {tsl['origQty']}, Trailing Delta: {delta_percent:.2f}% (Activation Price: ${activate_price:,.2f} NOT YET HIT)")
                        if pd.notna(stop_price_from_api) and stop_price_from_api != 0:
                            effective_sl_price = stop_price_from_api
                            loss_from_entry_at_sl = (entry_price - effective_sl_price) * tsl['origQty'] * (1 if position_side == 'LONG' else -1)
                            sign = '+' if loss_from_entry_at_sl < 0 else '-'
                            print(f"    - Worst Case Trigger: ${effective_sl_price:,.4f} -> {sign}${abs(loss_from_entry_at_sl):,.2f}")
                            if abs(loss_from_entry_at_sl) > abs(coin_max_loss):
                                coin_max_loss = loss_from_entry_at_sl
                        else:
                            print("    - No fixed stopPrice provided before activation. Risk is undefined for this portion.")
                    else:
                        if position_side == 'LONG':
                            effective_sl_price = mark_price * (1 - delta_percent / 100)
                        else:
                            effective_sl_price = mark_price * (1 + delta_percent / 100)

                        loss_from_entry_at_sl = (entry_price - effective_sl_price) * tsl['origQty'] * (1 if position_side == 'LONG' else -1)
                        sign = '+' if loss_from_entry_at_sl < 0 else '-'
                        print(f"  - Type: {tsl['type']}, Quantity: {tsl['origQty']}, Trailing Delta: {delta_percent:.2f}% (ACTIVE)")
                        print(f"    - Current Worst Case Trigger: ${effective_sl_price:,.4f} -> {sign}${abs(loss_from_entry_at_sl):,.2f}")
                        if abs(loss_from_entry_at_sl) > abs(coin_max_loss):
                            coin_max_loss = loss_from_entry_at_sl

            total_sl_covered_qty = sl_orders['origQty'].sum() + trailing_sl_orders['origQty'].sum()
            uncovered_qty = abs(position_amt) - total_sl_covered_qty

            if uncovered_qty > 0:
                uncovered_value_usdt = uncovered_qty * mark_price
                print(f"  - ⚠️ CAUTION: No Stop-Loss (fixed or active trailing) for {uncovered_qty} units (approx. ${uncovered_value_usdt:,.2f}). Risk is undefined.")

            net_max_loss_from_stops += max(0, coin_max_loss)
            sign = '-' if coin_max_loss > 0 else '+'
            print(f"  - Worst-Case PnL for {symbol} (if SL triggers): {sign}${abs(coin_max_loss):,.2f}\n")

            tp_orders = associated_orders[associated_orders['type'].isin(['TAKE_PROFIT_MARKET', 'TAKE_PROFIT', 'TAKE_PROFIT_LIMIT'])]
            tp_quantity_covered = tp_orders['origQty'].sum()
            uncovered_tp_qty = abs(position_amt) - tp_quantity_covered

            position_max_profit = 0
            if not tp_orders.empty:
                for _, tp in tp_orders.iterrows():
                    if pd.notna(tp['stopPrice']):
                        profit = abs(tp['stopPrice'] - entry_price) * tp['origQty']
                        position_max_profit += profit
                total_potential_profit += position_max_profit
                print(f"  - Max Profit (from TP): +${position_max_profit:,.2f} across {tp_quantity_covered} units.")

            if uncovered_tp_qty > 0:
                print(f"  - No Take-Profit for {uncovered_tp_qty} units.\n")

            limit_orders = associated_orders[associated_orders['type'] == 'LIMIT']
            if not limit_orders.empty:
                print(f"🔵 UNFILLED LIMIT ORDERS:")
                for _, order in limit_orders.iterrows():
                    order_value = order['origQty'] * order['price']
                    print(f"  - {order['side']} {order['origQty']} at ${order['price']:,.2f} (Value: ${order_value:,.2f})")
                print("    (Note: P/L for these orders can only be calculated after they are filled and a position is opened.)")

            print(f"===============================================\n")

        print("\n--- 🥅 NET ACCOUNT SUMMARY 🥅 ---")
        sign_pnl = '+' if total_current_unrealized_pnl >= 0 else '-'
        print(f"Total Current Unrealized PnL:    {sign_pnl}${abs(total_current_unrealized_pnl):,.2f}")
        print(f"Total Potential Profit (from all TPs): +${total_potential_profit:,.2f}")
        print(f"Total Net Max Loss (from all SLs): -${net_max_loss_from_stops:,.2f}")
        print("-----------------------------------")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure your API keys are correct and have Futures permissions.")

if __name__ == "__main__":
    analyze_futures_account()
