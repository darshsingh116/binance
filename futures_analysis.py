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
    orders, potential profits/losses, and warnings for missing SL/TP and Trailing SL.
    Includes current unrealized PnL for each coin.
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
        # 'unRealizedProfit' is already here, just ensuring it's numeric
        for df in [positions_df, orders_df]:
            if df.empty: continue
            for col in ['positionAmt', 'entryPrice', 'markPrice', 'unRealizedProfit', 'origQty', 'price', 'stopPrice', 'priceRate', 'activatePrice']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    df[col] = pd.NA

        total_potential_profit = 0
        total_potential_loss = 0
        total_current_unrealized_pnl = 0 # To sum up all current PnLs

        # --- Get unique symbols from open positions ---
        unique_symbols = positions_df['symbol'].unique()

        print("--- Account Risk & Profit Analysis ---\n")

        for symbol in unique_symbols:
            print(f"==================== {symbol} ====================")
            print(f"  - Note: 'Min Profit/Loss' for active trailing stops is based on current mark price and callback rate, representing worst-case from *this point*.")
            print(f"  - Note: 'Unrealized PnL' is the current profit/loss if you closed the position immediately.\n")

            
            # --- Position Info ---
            position = positions_df[positions_df['symbol'] == symbol].iloc[0]
            position_amt = position['positionAmt']
            entry_price = position['entryPrice']
            mark_price = position['markPrice']
            position_side = 'LONG' if position_amt > 0 else 'SHORT'
            position_value_usdt = abs(position_amt * mark_price)
            unrealized_profit = position['unRealizedProfit'] # Get the current unrealized PnL
            total_current_unrealized_pnl += unrealized_profit

            print(f"✅ OPEN POSITION:")
            print(f"  - Side: {position_side}, Quantity: {position_amt}, Value: ${position_value_usdt:,.2f}")
            print(f"  - Entry Price: ${entry_price:,.4f}, Mark Price: ${mark_price:,.4f}")
            print(f"  - Current Unrealized PnL: ${unrealized_profit:,.2f}\n") # Display current PnL

            # --- Find associated orders for this symbol ---
            associated_orders = orders_df[orders_df['symbol'] == symbol] if not orders_df.empty else pd.DataFrame()
            
            # --- Stop Loss Analysis (Regular SL, excluding Trailing SL) ---
            sl_orders = associated_orders[
                associated_orders['type'].isin(['STOP_MARKET', 'STOP_LOSS', 'STOP_LOSS_LIMIT']) &
                (associated_orders['type'] != 'TRAILING_STOP_MARKET')
            ]
            
            sl_quantity_covered = sl_orders['origQty'].sum()
            
            position_max_loss_from_sl = 0
            if not sl_orders.empty:
                for _, sl in sl_orders.iterrows():
                    if pd.notna(sl['stopPrice']):
                        # Calculate potential loss for regular SL
                        # Note: This is an *absolute* loss from entry, not relative to current PnL
                        loss = abs(entry_price - sl['stopPrice']) * sl['origQty']
                        position_max_loss_from_sl += loss
                total_potential_loss += position_max_loss_from_sl
                print(f"  - Max Loss (from Fixed SL): ${position_max_loss_from_sl:,.2f} across {sl_quantity_covered} units.")
            
            # --- Trailing Stop Loss Analysis ---
            trailing_sl_orders = associated_orders[associated_orders['type'] == 'TRAILING_STOP_MARKET']
            
            uncovered_qty_by_trailing_sl = 0 # Quantity for which a specific stop is not active or set
            if not trailing_sl_orders.empty:
                print(f"🟠 TRAILING STOP LOSS ORDERS:")
                for _, tsl in trailing_sl_orders.iterrows():
                    delta_percent = tsl['priceRate']
                    activate_price = tsl['activatePrice']
                    stop_price_from_api = tsl['stopPrice'] # The static stopPrice from API

                    # Determine the effective "current" stop loss price
                    effective_sl_price = None
                    profit_at_sl = None

                    # Use position_side to correctly determine if activatePrice has been hit
                    if position_side == 'LONG':
                        activation_condition_met = pd.isna(activate_price) or mark_price >= activate_price
                    else: # SHORT position
                        activation_condition_met = pd.isna(activate_price) or mark_price <= activate_price

                    if not activation_condition_met: # If activatePrice is set and not yet hit
                        print(f"  - Type: {tsl['type']}, Quantity: {tsl['origQty']}, Trailing Delta: {delta_percent:.2f}% (Activation Price: ${activate_price:,.2f} NOT YET HIT)")
                        if pd.notna(stop_price_from_api) and stop_price_from_api != 0:
                            effective_sl_price = stop_price_from_api
                            profit_at_sl = (effective_sl_price - entry_price) * tsl['origQty'] * (1 if position_side == 'LONG' else -1)
                            print(f"    - Current Worst Case Trigger: ${effective_sl_price:,.4f} -> Min Profit/Loss: ${profit_at_sl:,.2f}")
                        else:
                            print("    - No fixed stopPrice provided before activation. Risk is undefined for this portion.")
                            uncovered_qty_by_trailing_sl += tsl['origQty'] # This portion is "uncovered" until activation
                    else: # Trailing stop is active (either activatePrice hit, or not set)
                        # Calculate the dynamic worst-case trigger price from the current markPrice
                        if position_side == 'LONG':
                            effective_sl_price = mark_price * (1 - delta_percent / 100)
                        else: # SHORT position
                            effective_sl_price = mark_price * (1 + delta_percent / 100)
                        
                        profit_at_sl = (effective_sl_price - entry_price) * tsl['origQty'] * (1 if position_side == 'LONG' else -1)
                        # Add to total_potential_loss only if it represents a *new* or increased loss compared to current PnL.
                        # For simplicity here, we're adding the absolute value of the calculated loss from entry.
                        # A more precise total_potential_loss would require comparing against unrealized_profit.
                        total_potential_loss += abs(profit_at_sl) if profit_at_sl < 0 else 0 

                        print(f"  - Type: {tsl['type']}, Quantity: {tsl['origQty']}, Trailing Delta: {delta_percent:.2f}% (ACTIVE)")
                        print(f"    - Current Worst Case Trigger: ${effective_sl_price:,.4f} -> Min Profit/Loss: ${profit_at_sl:,.2f}")

            # Calculate total uncovered quantity including those not covered by any active stop loss
            # This logic needs to consider quantities covered by fixed SLs and those that have active trailing stops.
            # A more robust calculation would track covered quantity across all order types.
            # For now, let's keep it simple: total position quantity minus quantity covered by fixed SLs.
            # The 'uncovered_qty_by_trailing_sl' is for situations where TSL isn't active yet and no fixed SL is present.
            remaining_position_qty = abs(position_amt) - sl_quantity_covered
            
            # Only report 'uncovered' if no active trailing stop or fixed stop for that portion
            # This part needs careful thought to avoid double-counting or missing coverage.
            # The simplest way is to check if total position quantity is fully covered by *any* kind of stop.
            total_stops_quantity = sl_quantity_covered + trailing_sl_orders['origQty'].sum() # Simple sum, might need refinement for overlaps
            
            uncovered_total_position_qty = abs(position_amt) - total_stops_quantity
            
            if uncovered_total_position_qty > 0:
                 uncovered_value_usdt = uncovered_total_position_qty * mark_price
                 print(f"  - ⚠️ CAUTION: No Stop-Loss (fixed or active trailing) for {uncovered_total_position_qty} units (approx. ${uncovered_value_usdt:,.2f}).")


           

            # --- Take Profit Analysis ---
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
                print(f"  - Max Profit (from TP): ${position_max_profit:,.2f} across {tp_quantity_covered} units.")

            if uncovered_tp_qty > 0:
                print(f"  - No Take-Profit for {uncovered_tp_qty} units.\n")


            # --- Unfilled Limit Order Analysis ---
            limit_orders = associated_orders[associated_orders['type'] == 'LIMIT']
            if not limit_orders.empty:
                print(f"🔵 UNFILLED LIMIT ORDERS:")
                for _, order in limit_orders.iterrows():
                    order_value = order['origQty'] * order['price']
                    print(f"  - {order['side']} {order['origQty']} at ${order['price']:,.2f} (Value: ${order_value:,.2f})")
                print("    (Note: P/L for these orders can only be calculated after they are filled and a position is opened.)")

            print(f"===============================================\n")


        # --- Final Summary ---
        print("\n--- 🥅 NET ACCOUNT SUMMARY 🥅 ---")
        print(f"Total Current Unrealized PnL:     ${total_current_unrealized_pnl:,.2f}") # New summary line
        print(f"Total Potential Profit (from all TPs): ${total_potential_profit:,.2f}")
        print(f"Total Potential Loss (from all SLs):   ${total_potential_loss:,.2f}")
        print("-----------------------------------")


    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure your API keys are correct and have Futures permissions.")

if __name__ == "__main__":
    analyze_futures_account()