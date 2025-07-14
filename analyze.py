import os
import pandas as pd
from dotenv import load_dotenv
from binance.client import Client

load_dotenv("key.env")
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

client = Client(API_KEY, SECRET_KEY)

def get_futures_analysis():
    result = {
        "per_coin_data": [],
        "total_unrealized_pnl": 0,
        "net_max_loss_from_stops": 0,
        "total_potential_profit": 0
    }

    try:
        positions = client.futures_position_information()
        open_orders = client.futures_get_open_orders()

        open_positions = [p for p in positions if float(p['positionAmt']) != 0]
        if not open_positions:
            return result

        positions_df = pd.DataFrame(open_positions)
        orders_df = pd.DataFrame(open_orders) if open_orders else pd.DataFrame()

        for df in [positions_df, orders_df]:
            if df.empty: continue
            for col in ['positionAmt', 'entryPrice', 'markPrice', 'unRealizedProfit', 'origQty', 'price', 'stopPrice', 'priceRate', 'activatePrice']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    df[col] = pd.NA

        for symbol in positions_df['symbol'].unique():
            position = positions_df[positions_df['symbol'] == symbol].iloc[0]
            position_amt = position['positionAmt']
            entry_price = position['entryPrice']
            mark_price = position['markPrice']
            position_side = 'LONG' if position_amt > 0 else 'SHORT'
            unrealized_profit = position['unRealizedProfit']
            position_value = abs(position_amt * mark_price)
            result['total_unrealized_pnl'] += unrealized_profit

            associated_orders = orders_df[orders_df['symbol'] == symbol] if not orders_df.empty else pd.DataFrame()
            sl_orders = associated_orders[
                associated_orders['type'].isin(['STOP_MARKET', 'STOP_LOSS', 'STOP_LOSS_LIMIT']) &
                (associated_orders['type'] != 'TRAILING_STOP_MARKET')
            ]

            trailing_sl_orders = associated_orders[associated_orders['type'] == 'TRAILING_STOP_MARKET']
            tp_orders = associated_orders[associated_orders['type'].isin(['TAKE_PROFIT_MARKET', 'TAKE_PROFIT', 'TAKE_PROFIT_LIMIT'])]
            limit_orders = associated_orders[associated_orders['type'] == 'LIMIT']

            coin_max_loss = 0
            sl_details = []
            tp_details = []
            limit_details = []

            for _, sl in sl_orders.iterrows():
                if pd.notna(sl['stopPrice']):
                    loss_from_entry = abs(entry_price - sl['stopPrice']) * sl['origQty']
                    if position_side == 'LONG' and sl['stopPrice'] >= entry_price:
                        loss_value = -loss_from_entry
                    elif position_side == 'SHORT' and sl['stopPrice'] <= entry_price:
                        loss_value = -loss_from_entry
                    else:
                        loss_value = loss_from_entry
                    if abs(loss_value) > abs(coin_max_loss):
                        coin_max_loss = loss_value
                    sl_details.append({
                        "type": sl["type"],
                        "qty": sl["origQty"],
                        "stopPrice": sl["stopPrice"],
                        "loss": round(loss_value, 2)
                    })

            for _, tsl in trailing_sl_orders.iterrows():
                if pd.notna(tsl['priceRate']) and pd.notna(tsl['origQty']):
                    delta = tsl['priceRate']
                    if position_side == 'LONG':
                        effective_sl_price = mark_price * (1 - delta / 100)
                    else:
                        effective_sl_price = mark_price * (1 + delta / 100)
                    loss_from_entry = (entry_price - effective_sl_price) * tsl['origQty'] * (1 if position_side == 'LONG' else -1)
                    if abs(loss_from_entry) > abs(coin_max_loss):
                        coin_max_loss = loss_from_entry
                    sl_details.append({
                        "type": "TRAILING_STOP_MARKET",
                        "qty": tsl["origQty"],
                        "stopPrice": round(effective_sl_price, 4),
                        "loss": round(loss_from_entry, 2)
                    })

            uncovered_sl_qty = abs(position_amt) - sum(d['qty'] for d in sl_details)
            uncovered_tp_qty = abs(position_amt)

            tp_profit_total = 0
            for _, tp in tp_orders.iterrows():
                if pd.notna(tp['stopPrice']):
                    profit = abs(tp['stopPrice'] - entry_price) * tp['origQty']
                    tp_profit_total += profit
                    tp_details.append({
                        "qty": tp["origQty"],
                        "target": tp["stopPrice"],
                        "profit": round(profit, 2)
                    })
                    uncovered_tp_qty -= tp["origQty"]

            for _, lo in limit_orders.iterrows():
                order_value = lo['origQty'] * lo['price']
                limit_details.append({
                    "side": lo["side"],
                    "qty": lo["origQty"],
                    "price": lo["price"],
                    "value": round(order_value, 2)
                })

            result["net_max_loss_from_stops"] += max(0, coin_max_loss)
            result["total_potential_profit"] += tp_profit_total

            result["per_coin_data"].append({
                "symbol": symbol,
                "side": position_side,
                "position_amt": position_amt,
                "entry_price": entry_price,
                "mark_price": mark_price,
                "unrealized_pnl": unrealized_profit,
                "position_value": round(position_value, 2),
                "sl_orders": sl_details,
                "tp_orders": tp_details,
                "limit_orders": limit_details,
                "uncovered_sl_qty": uncovered_sl_qty,
                "uncovered_tp_qty": uncovered_tp_qty,
                "max_loss": coin_max_loss
            })

        return result

    except Exception as e:
        return {"error": str(e)}
