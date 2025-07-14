import pandas as pd
from binance.client import Client
import os
from dotenv import load_dotenv

# --- Load API Keys from .env file ---
load_dotenv("key.env")
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

# --- Initialize Binance Client ---
# Handles potential time sync issues automatically
client = Client(API_KEY, SECRET_KEY)

def analyze_futures_account():
    """
    Fetches and provides a detailed per-coin analysis of futures positions,
    orders, potential profits/losses, and warnings for missing SL/TP.
    """
    try:
        # --- Fetch All Positions and Open Orders ---
        positions = client.futures_position_information()
        open_orders = client.futures_get_open_orders()

        # --- Filter for actual open positions ---
        open_positions = [p for p in positions if float(p['positionAmt']) != 0]

        if not open_positions:
            print("No open futures positions found.")
            return

        # --- Create DataFrames for easier manipulation ---
        positions_df = pd.DataFrame(open_positions)
        orders_df = pd.DataFrame(open_orders) if open_orders else pd.DataFrame()

        # --- Convert relevant columns to numeric types for calculation ---
        for df in [positions_df, orders_df]:
            if df.empty: continue
            for col in ['positionAmt', 'entryPrice', 'markPrice', 'unRealizedProfit', 'origQty', 'price', 'stopPrice']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        total_potential_profit = 0
        total_potential_loss = 0

        # --- Get unique symbols from open positions ---
        unique_symbols = positions_df['symbol'].unique()

        print("--- Account Risk & Profit Analysis ---\n")

        for symbol in unique_symbols:
            print(f"==================== {symbol} ====================")
            
            # --- Position Info ---
            position = positions_df[positions_df['symbol'] == symbol].iloc[0]
            position_amt = position['positionAmt']
            entry_price = position['entryPrice']
            mark_price = position['markPrice']
            position_side = 'LONG' if position_amt > 0 else 'SHORT'
            position_value_usdt = abs(position_amt * mark_price)

            print(f"✅ OPEN POSITION:")
            print(f"   - Side: {position_side}, Quantity: {position_amt}, Value: ${position_value_usdt:,.2f}\n")

            # --- Find associated orders for this symbol ---
            associated_orders = orders_df[orders_df['symbol'] == symbol] if not orders_df.empty else pd.DataFrame()
            
            # --- Stop Loss Analysis ---
            sl_orders = associated_orders[associated_orders['type'] == 'STOP_MARKET']
            sl_quantity_covered = sl_orders['origQty'].sum()
            uncovered_sl_qty = abs(position_amt) - sl_quantity_covered
            
            position_max_loss = 0
            if not sl_orders.empty:
                for _, sl in sl_orders.iterrows():
                    loss = abs(entry_price - sl['stopPrice']) * sl['origQty']
                    position_max_loss += loss
                total_potential_loss += position_max_loss
                print(f"   - Max Loss (from SL): ${position_max_loss:,.2f} across {sl_quantity_covered} units.")
            
            if uncovered_sl_qty > 0:
                uncovered_value_usdt = uncovered_sl_qty * mark_price
                print(f"   - ⚠️ CAUTION: No Stop-Loss for {uncovered_sl_qty} units (approx. ${uncovered_value_usdt:,.2f}).")

            # --- Take Profit Analysis ---
            tp_orders = associated_orders[associated_orders['type'] == 'TAKE_PROFIT_MARKET']
            tp_quantity_covered = tp_orders['origQty'].sum()
            uncovered_tp_qty = abs(position_amt) - tp_quantity_covered
            
            position_max_profit = 0
            if not tp_orders.empty:
                for _, tp in tp_orders.iterrows():
                    profit = abs(tp['stopPrice'] - entry_price) * tp['origQty']
                    position_max_profit += profit
                total_potential_profit += position_max_profit
                print(f"   - Max Profit (from TP): ${position_max_profit:,.2f} across {tp_quantity_covered} units.")

            if uncovered_tp_qty > 0:
                print(f"   - No Take-Profit for {uncovered_tp_qty} units.\n")


            # --- Unfilled Limit Order Analysis ---
            limit_orders = associated_orders[associated_orders['type'] == 'LIMIT']
            if not limit_orders.empty:
                print(f"🔵 UNFILLED LIMIT ORDERS:")
                for _, order in limit_orders.iterrows():
                     order_value = order['origQty'] * order['price']
                     print(f"   - {order['side']} {order['origQty']} at ${order['price']:,.2f} (Value: ${order_value:,.2f})")
                print("     (Note: P/L for these orders can only be calculated after they are filled and a position is opened.)")

            print(f"===============================================\n")


        # --- Final Summary ---
        print("\n--- 🥅 NET ACCOUNT SUMMARY 🥅 ---")
        print(f"Total Potential Profit (from all TPs): ${total_potential_profit:,.2f}")
        print(f"Total Potential Loss (from all SLs):   ${total_potential_loss:,.2f}")
        print("-----------------------------------")


    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure your API keys are correct and have Futures permissions.")

if __name__ == "__main__":
    analyze_futures_account()