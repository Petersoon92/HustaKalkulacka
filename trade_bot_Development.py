import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.models import load_model
import time
import os
import warnings
import re
import shutil

MODEL_DIRECTORY = "C:/Users/petrk/Desktop/ML's projekt/SavedData"
BACKTEST_RESULTS_FILE = "C:/Users/petrk/Desktop/ML's projekt/SavedData/best_backtest_results.csv"
MODEL_NAME = "trade_bot_model.h5"  # This is your original model name without numbers
MODEL_PATH = os.path.join(MODEL_DIRECTORY, MODEL_NAME)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class EpochEndCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1} completed.")

ORDER_BUY = 0
ORDER_SELL = 1

opened_positions = {"BUY": None, "SELL": None}
total_profit = 0.0
total_trades = 0
successful_trades = 0
unsuccessful_trades = 0

import datetime

def fetch_mt5_data(symbol, bars=10000):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, bars)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df[['close', 'tick_volume']]
    df.columns = ['Adj Close', 'Volume']
    df.to_csv('C:/Users/petrk/Desktop/ML\'s projekt/SavedData/mt5_data.csv')
    return df

def prepare_data(df, lookback):
    features = ['Adj Close', 'Volume']
    dataset = df[features].copy()
    scalers = {}
    scaled_data = {}

    for feature in features:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data[feature] = scaler.fit_transform(dataset[feature].values.reshape(-1, 1)).flatten()
        scalers[feature] = scaler

    dataset = pd.DataFrame(scaled_data)
    x_data, y_data = [], []

    for i in range(lookback, len(dataset)):
        x_data.append(dataset.iloc[i - lookback:i].values)
        y_data.append(dataset['Adj Close'].iloc[i])

    x_data, y_data = np.array(x_data), np.array(y_data)
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], len(features)))

    return x_data, y_data, scalers

def build_lstm_model(lookback, feature_count):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, feature_count)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def trading_decision(predicted_value, current_value):
    if predicted_value > current_value:
        return ORDER_BUY
    elif predicted_value < current_value:
        return ORDER_SELL
    else:
        return None

def close_order(ticket):
    global total_trades, successful_trades, unsuccessful_trades
    """Close an open position by ticket number."""
    positions = mt5.positions_get(ticket=ticket)
    
    if not positions or len(positions) == 0:
        print(f"Failed to get position with ticket number {ticket}")
        return False, 0.0  # Return 0.0 profit for this case

    position = positions[0]
    
    action = ORDER_SELL if position.type == ORDER_BUY else ORDER_BUY
    price = mt5.symbol_info_tick(position.symbol).ask if action == ORDER_BUY else mt5.symbol_info_tick(position.symbol).bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": action,
        "price": price,
        "deviation": 10,
        "magic": 234000,
        "comment": f"Closing {ticket}",
        "position": position.ticket,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        total_trades += 1
        profit = position.profit  # Get the profit from the closed position
        if profit > 0:
            successful_trades += 1
        else:
            unsuccessful_trades += 1
        print(f"Closed position with ticket number {ticket}")
        return True, profit
    else:
        print(f"Failed to close position with ticket number {ticket}, error:", mt5.last_error())
        return False, 0.0  # Return 0.0 profit for this case


def place_order(action, symbol, volume):
    global opened_positions
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if action == ORDER_BUY else tick.bid
    action_str = "BUY" if action == ORDER_BUY else "SELL"

    opposite_action = "SELL" if action == ORDER_BUY else "BUY"
    if opened_positions[opposite_action]:
        if close_order(opened_positions[opposite_action]):
            opened_positions[opposite_action] = None

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": action,
        "price": price,
        "deviation": 10,
        "magic": 234000,
        "comment": "Python script order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"Order {action_str} executed at {price} with ticket number {result.order}")
        opened_positions[action_str] = result.order
        return True
    else:
        print("Order failed, error:", mt5.last_error())
        return False

def calculate_unrealized_profit():
    """Calculate the total unrealized profit for all open positions."""
    positions = mt5.positions_get()
    
    if not positions:
        return 0.0

    unrealized_profit = sum(position.profit for position in positions)
    return unrealized_profit

def backtest(model, data, lookback, scalers):
    initial_balance = 10000.0  # starting balance
    balance = initial_balance
    position = None  # track if we're holding a buy/sell position
    position_price = 0.0  # track the price we entered the position
    
    backtest_successful = 0
    backtest_unsuccessful = 0

    for i in range(lookback, len(data) - 1):  # -1 because we predict next period
        current_df = data.iloc[i - lookback:i + 1]
        x_latest, _, _ = prepare_data(current_df, lookback)
        scaled_prediction = model.predict(x_latest[-1].reshape(1, lookback, 2), verbose=0)
        prediction = scalers['Adj Close'].inverse_transform(scaled_prediction)

        current_value = current_df['Adj Close'].iloc[-1]
        action = trading_decision(prediction[0][0], current_value)

        # Handle buying logic
        if action == ORDER_BUY and position != ORDER_BUY:
            if position == ORDER_SELL:
                profit = position_price - current_value  # close sell position
                if profit > 0:
                    backtest_successful += 1
                else:
                    backtest_unsuccessful += 1
                balance += profit
            position_price = current_value  # buy at current value
            position = ORDER_BUY

        # Handle selling logic
        elif action == ORDER_SELL and position != ORDER_SELL:
            if position == ORDER_BUY:
                profit = current_value - position_price  # close buy position
                if profit > 0:
                    backtest_successful += 1
                else:
                    backtest_unsuccessful += 1
                balance += profit
            position_price = current_value  # sell at current value
            position = ORDER_SELL

    # Finalize the last open position at the end of the data
    if position == ORDER_BUY:
        profit = data['Adj Close'].iloc[-1] - position_price
        if profit > 0:
            backtest_successful += 1
        else:
            backtest_unsuccessful += 1
        balance += profit
    elif position == ORDER_SELL:
        profit = position_price - data['Adj Close'].iloc[-1]
        if profit > 0:
            backtest_successful += 1
        else:
            backtest_unsuccessful += 1
        balance += profit

    # Calculate the average profit per trade
    total_trades_backtest = backtest_successful + backtest_unsuccessful
    if total_trades_backtest > 0:
        avg_profit_per_trade = (balance - initial_balance) / total_trades_backtest
        profit_ratio = backtest_successful / total_trades_backtest
    else:
        avg_profit_per_trade = 0
        profit_ratio = 0  # Handle cases where no trades were made to avoid division by zero

    return balance - initial_balance, backtest_successful, backtest_unsuccessful

def get_highest_model_number(path):
    """Get the highest model number from the saved models in the directory."""
    files = os.listdir(path)
    model_numbers = []
    pattern = re.compile(r'\d+$')
    for file in files:
        if MODEL_NAME in file:
            match = pattern.search(file)
            if match:
                model_numbers.append(int(match.group()))
    return max(model_numbers, default=0)

def get_next_model_name(path):
    """Get the next model name by incrementing the highest model number."""
    highest_number = get_highest_model_number(path)
    next_number = highest_number + 1
    return f"{MODEL_NAME}_{next_number}"

def get_latest_model_version():
    """
    Get the latest model version number from the saved models.
    """
    dirs = [d for d in os.listdir(MODEL_DIRECTORY) if os.path.isdir(os.path.join(MODEL_DIRECTORY, d)) and d.startswith("trade_bot_model_")]

    # Extract version numbers from the model directory names
    versions = [int(d.split('_')[-1]) for d in dirs]

    if versions:
        return max(versions)
    else:
        return None

BACKTEST_RESULTS_FILE = "C:/Users/petrk/Desktop/ML's projekt/SavedData/best_backtest_results.csv"

def save_best_backtest_results(profit):
    with open(BACKTEST_RESULTS_FILE, 'w') as file:
        file.write(str(profit))

def load_best_backtest_results():
    if os.path.exists(BACKTEST_RESULTS_FILE):
        with open(BACKTEST_RESULTS_FILE, 'r') as file:
            return float(file.readline().strip())
    return float('-inf')

def update_model_directory():
    # Get the latest model version
    latest_model_version = get_latest_model_version()

    # Create a new model directory with an incremented version number
    next_model_version = latest_model_version + 1 if latest_model_version is not None else 1
    next_model_dir = os.path.join(MODEL_DIRECTORY, f"trade_bot_model_{next_model_version}")


    return next_model_dir, next_model_version
    
def rename_existing_model_directory(current_version):
    # For trade_bot_model
    current_trade_model_name = f"trade_bot_model_{current_version}"
    current_trade_model_path = os.path.join(MODEL_DIRECTORY, current_trade_model_name)

    next_version = current_version + 1
    next_trade_model_name = f"trade_bot_model_{next_version}"
    next_trade_model_path = os.path.join(MODEL_DIRECTORY, next_trade_model_name)

    if os.path.exists(current_trade_model_path):
        os.rename(current_trade_model_path, next_trade_model_path)

    # For backtest_bot_model
    current_backtest_model_name = f"backtest_bot_model_{current_version}"
    current_backtest_model_path = os.path.join(MODEL_DIRECTORY, current_backtest_model_name)

    next_backtest_model_name = f"backtest_bot_model_{next_version}"
    next_backtest_model_path = os.path.join(MODEL_DIRECTORY, next_backtest_model_name)

    if os.path.exists(current_backtest_model_path):
        os.rename(current_backtest_model_path, next_backtest_model_path)

    return next_trade_model_path, next_version  # returning trade_bot path just to keep previous functionality


def save_backtest_time(duration):
    path = "C:/Users/petrk/Desktop/ML's projekt/SavedData/backtest_duration.csv"
    df = pd.DataFrame([duration], columns=["duration"])
    df.to_csv(path, index=False)

def load_backtest_time():
    path = "C:/Users/petrk/Desktop/ML's projekt/SavedData/backtest_duration.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df['duration'][0]
    return None

def main():
    total_bars = 2000  # Total number of bars to keep in memory.
    train_size = 1500  # The size of the training dataset.
    global opened_positions, total_profit
    
    # Define early_stopping here, before any if/else logic
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Initialize backtest_counter and other variables
    backtest_counter = 1
    successful_trades = 0
    unsuccessful_trades = 0
    avg_profit = 0.0
    profit_ratio_percentage = 0.0
    best_backtest_profit = load_best_backtest_results()

    if not mt5.initialize():
        print("MetaTrader5 initialization failed!")
        return
    print("MetaTrader5 initialized successfully!")

    symbol = "EURUSD"
    volume = 0.1
    lookback = 48
    last_action = None

    df = fetch_mt5_data(symbol, bars=total_bars)
    train_data = df.iloc[:train_size]
    backtest_data = df.iloc[train_size:]
    x_train, y_train, scalers = prepare_data(train_data, lookback)
    initial_training_size = len(train_data)

    if len(x_train) < 2:
        print("Not enough data for training. Exiting.")
        return

    latest_model_version = get_latest_model_version()
    if latest_model_version is not None:
        latest_model_path = os.path.join(MODEL_DIRECTORY, f"trade_bot_model_{latest_model_version}")
        if os.path.exists(latest_model_path):
            model = tf.keras.models.load_model(latest_model_path)
            print("Loaded previously trained model.")
    else:
        model = build_lstm_model(lookback, x_train.shape[2])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(x_train, y_train, batch_size=1, epochs=10, 
                  callbacks=[early_stopping, EpochEndCallback()], verbose=0)
        new_model_dir, latest_model_version = update_model_directory()
        backtest_model_dir = new_model_dir.replace("trade_bot_model", "backtest_bot_model")
        new_model_path = os.path.join(MODEL_DIRECTORY, f"trade_bot_model_{latest_model_version}")

        backtest_model_path = new_model_path.replace("trade_bot_model", "backtest_bot_model")
        model.save(new_model_path, save_format='tf')
        model.save(backtest_model_path, save_format='tf')
        print(f"Model saved as trade bot in {new_model_dir} and as backtest bot in {backtest_model_dir}.")
        print("Model trained and saved for the first time.")
        
        # Immediate backtest after training the initial model
        print("Performing backtest on the initial model...")
        profit_or_loss, backtest_successful_trades, backtest_unsuccessful_trades = backtest(model, backtest_data, lookback, scalers)
        total_trades_backtest = backtest_successful_trades + backtest_unsuccessful_trades
        if total_trades_backtest > 0:
            avg_profit = profit_or_loss / total_trades_backtest
            profit_ratio_percentage = (backtest_successful_trades / total_trades_backtest) * 100
        print(f"Backtest Result on the initial model: {profit_or_loss:.2f}")
        print(f"Successful Trades during backtest: {backtest_successful_trades}")
        print(f"Unsuccessful Trades during backtest: {backtest_unsuccessful_trades}")
        print(f"Average Profit per Trade during backtest: {avg_profit:.2f}")
        print(f"Profit Ratio during backtest: {profit_ratio_percentage:.2f}%")
        print("Backtest on initial model completed.")
        best_backtest_profit = profit_or_loss  # Set the initial model's profit as the best
        
        # Fine-tune trade_bot on the rest of the data
        x_all, y_all, _ = prepare_data(df, lookback)
        num_bars = len(x_all) - initial_training_size + lookback
        model.fit(x_all, y_all, batch_size=1, epochs=10, 
                  callbacks=[early_stopping, EpochEndCallback()], verbose=0)
        new_model_path = os.path.join(MODEL_DIRECTORY, f"trade_bot_model_{latest_model_version}")
        model.save(new_model_path, save_format='tf')
        print(f"Model fine-tuned on additional {num_bars} bars and saved at {new_model_path}")


    prediction_interval = 60 * 15
    last_prediction_time = 0
    first_prediction_done = False

    while True:
        current_time = time.time()
        if current_time - last_prediction_time >= prediction_interval:
            df_latest = fetch_mt5_data(symbol, bars=total_bars)  # Fetch more bars
            new_data = df_latest[df_latest.index > df.index[-1]]
            if not new_data.empty:
                df = pd.concat([df.tail(total_bars - len(new_data)), new_data])  # Keep the dataframe size consistent

            x_latest, _, _ = prepare_data(df_latest.tail(lookback + 1), lookback)
            
            new_bars_added = len(new_data)
            total_bars_available = len(df)
            print(f"{new_bars_added} new bars added since last prediction. Total bars available: {total_bars}.")
    
            # Add the latest data to your existing dataset and prepare it for retraining
            df = pd.concat([df, df_latest.tail(1)])
            x_train, y_train, scalers = prepare_data(df, lookback)
            
            # Fine-tune the trading model on the entire dataset after backtesting shows improvement
            x_all, y_all, _ = prepare_data(df, lookback)
            model.fit(x_all, y_all, batch_size=1, epochs=10, 
                      callbacks=[early_stopping, EpochEndCallback()], verbose=0)
            
            scaled_prediction = model.predict(x_latest[-1].reshape(1, lookback, 2))
            prediction = scalers['Adj Close'].inverse_transform(scaled_prediction)
            print(f"Predicted price for the next period: {prediction[0][0]}")

            current_value = df['Adj Close'].iloc[-1]
            action = trading_decision(prediction[0][0], current_value)

            if action is not None and action != last_action:
                opposite_action_str = "SELL" if action == ORDER_BUY else "BUY"
                if opened_positions[opposite_action_str]:
                    success, profit = close_order(opened_positions[opposite_action_str])
                    if success:
                        total_profit += profit
                        opened_positions[opposite_action_str] = None

                if place_order(action, symbol, volume):
                    last_action = action

            current_total_profit = total_profit + calculate_unrealized_profit()
            print(f"Total Profit/Loss: {current_total_profit:.2f}")
            print(f"Total Trades: {total_trades}")
            print(f"Successful Trades: {successful_trades}")
            print(f"Unsuccessful Trades: {unsuccessful_trades}")

            
            first_prediction_done = True
            last_prediction_time = current_time
        else:
            # Check if there's enough time for backtesting
            previous_backtest_duration = load_backtest_time()
            time_until_next_prediction = (last_prediction_time + prediction_interval) - current_time
            
            if previous_backtest_duration is None or time_until_next_prediction > previous_backtest_duration + 5:  # added +5 seconds buffer
                model = build_lstm_model(lookback, x_train.shape[2])
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                model.fit(x_train, y_train, batch_size=1, epochs=10,
                         callbacks=[early_stopping, EpochEndCallback()], verbose=0)

                profit_or_loss, backtest_successful_trades, backtest_unsuccessful_trades = backtest(model, backtest_data, lookback, scalers)
                
                total_trades_backtest = backtest_successful_trades + backtest_unsuccessful_trades
                if total_trades_backtest > 0:
                    avg_profit = profit_or_loss / total_trades_backtest
                    profit_ratio_percentage = (backtest_successful_trades / total_trades_backtest) * 100

                print(f"Backtest Result: {profit_or_loss:.2f}")
                print(f"Successful Trades during backtest: {backtest_successful_trades}")
                print(f"Unsuccessful Trades during backtest: {backtest_unsuccessful_trades}")
                print(f"Average Profit per Trade during backtest: {avg_profit:.2f}")
                print(f"Profit Ratio during backtest: {profit_ratio_percentage:.2f}%")
                
                if profit_or_loss > best_backtest_profit:
                    best_backtest_profit = profit_or_loss
                    save_best_backtest_results(best_backtest_profit)
                    print("Found a better model during backtesting. Replacing the trading model.")
                    if latest_model_version is None:
                        latest_model_version = 0
                    new_model_dir, latest_model_version = rename_existing_model_directory(latest_model_version)
                    new_model_path = os.path.join(MODEL_DIRECTORY, f"trade_bot_model_{latest_model_version}")
                    model.save(new_model_path, save_format='tf')
                    print(f"Model saved in {new_model_path}.")
                      
                    # Fine-tune the newly found better model on the most recent data
                    df_latest = fetch_mt5_data(symbol, bars=2000)  # Fetch more bars
                    df = pd.concat([df, df_latest])
                    x_all, y_all, _ = prepare_data(df, lookback)
                    num_bars = len(x_all) - initial_training_size + lookback
                    model.fit(x_all, y_all, batch_size=1, epochs=10, 
                              callbacks=[early_stopping, EpochEndCallback()], verbose=0)
                    model.save(new_model_path, save_format='tf')
                    num_new_bars = total_bars - train_size
                    print(f"Better model fine-tuned on additional {num_new_bars} bars and saved at {new_model_path}.")

                
                # Save this backtest duration
                backtest_end_time = time.time()
                backtest_duration = backtest_end_time - current_time
                save_backtest_time(backtest_duration)
            else:
                print("Skipping backtesting to allow time for prediction...")
                time.sleep(time_until_next_prediction)  # Sleep until it's time for the next prediction

        time.sleep(0)  # You can adjust the sleep time as required

if __name__ == "__main__":
    main()
