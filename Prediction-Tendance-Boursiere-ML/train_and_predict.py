import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import optuna
import warnings

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, log_loss

from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def to_series(x, index=None):
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    arr = np.asarray(x).reshape(-1,)
    return pd.Series(arr, index=index)

def rsi(close, window=14):
    diff = close.diff()
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def bollinger_bands(close, window=20, num_std=2):
    rolling_mean = close.rolling(window).mean()
    rolling_std = close.rolling(window).std()
    upper = rolling_mean + (rolling_std * num_std)
    lower = rolling_mean - (rolling_std * num_std)
    # Return normalized distance to bands (useful for ML)
    bb_width = (upper - lower) / rolling_mean
    bb_pos = (close - lower) / (upper - lower)
    return bb_width, bb_pos

def garman_klass_vol(open, high, low, close, window=20):
    # More efficient volatility estimator than standard deviation
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open) ** 2
    var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(var.rolling(window).mean())

#  DATA & FEATURES 

def load_data(ticker, start="2010-01-01"):
    df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError("No data found.")
    df = df.rename(columns=str.lower)
    # Flatten multi-index if present (yfinance update quirk)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    return df

def feature_engineering(df, horizon=1):
    out = df.copy()
    c = out['close']
    
    # 1. Targets
    # Predicting excess return vs volatility helps standardization
    ret_log = np.log(c).diff()
    out['target_ret'] = ret_log.shift(-horizon)
    out['y'] = (out['target_ret'] > 0).astype(int)
    
    # 2. Lagged Features (Crucial for autoregression)
    # Tells the model what happened 1, 2, 3, 5 days ago
    for lag in [1, 2, 3, 5, 10]:
        out[f'ret_lag_{lag}'] = ret_log.shift(lag-1)
        out[f'vol_lag_{lag}'] = np.log(out['volume']).diff().shift(lag-1)

    # 3. Technical Ratios (Normalized)
    # RSI
    out['rsi'] = rsi(c) / 100.0  # Scale 0-1
    
    
    for w in [20, 50, 200]:
        ma = c.rolling(w).mean()
        out[f'dist_ma{w}'] = (c / ma) - 1

   
    out['bb_width'], out['bb_pos'] = bollinger_bands(c)
    
    # Volatility Regime
    out['gk_vol'] = garman_klass_vol(out['open'], out['high'], out['low'], out['close'])
    out['vol_regime'] = out['gk_vol'] / out['gk_vol'].rolling(100).mean()

    # Interaction: Return / Volatility (Sharpe proxy)
    out['ret_vol_ratio'] = out['ret_lag_1'] / (out['gk_vol'] + 1e-6)

    # Seasonality
    out['day_of_week'] = out.index.dayofweek
    out['month'] = out.index.month

    out = out.dropna()
    
    features = [c for c in out.columns if c not in 
                ['open', 'high', 'low', 'close', 'volume', 'target_ret', 'y']]
    
    return out, features

# 3. OPTIMIZATION & TRAINING 

def objective_xgb(trial, X, y, train_idx, test_idx):
    """Optuna objective function for tuning XGBoost"""
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 2, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'n_jobs': -1,
        'random_state': 42,
        'eval_metric': 'logloss'
    }

    # Split for this specific fold calculation
    X_tr, X_val = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[test_idx]

    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr, verbose=False)
    
    preds = model.predict_proba(X_val)[:, 1]
    
    return roc_auc_score(y_val, preds)

def train_optimized_model(data, features, splits=5):
    X = data[features]
    y = data['y']
    
    # RobustScaler is better for financial data (outliers)
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features, index=X.index)

    tscv = TimeSeriesSplit(n_splits=splits, gap=2) # Gap prevents leakage
    
    all_preds = pd.Series(index=data.index, dtype=float)
    aucs = []

    print(f"Starting Walk-Forward Optimization ({splits} splits)...")
    
    final_params = {}

    for i, (tr, te) in enumerate(tscv.split(X_scaled)):
        print(f"  > Split {i+1}/{splits}: Tuning...", end="\r")
        
        # Optimize on the current training window
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective_xgb(t, X_scaled, y, tr, te), n_trials=15)
        
        best_params = study.best_params
        best_params.update({'n_jobs': -1, 'random_state': 42, 'eval_metric': 'logloss'})
        final_params = best_params # Save last for final model

        # Train with best params
        clf = XGBClassifier(**best_params)
        clf.fit(X_scaled.iloc[tr], y.iloc[tr])
        
        # Predict
        p = clf.predict_proba(X_scaled.iloc[te])[:, 1]
        all_preds.iloc[te] = p
        
        score = roc_auc_score(y.iloc[te], p)
        aucs.append(score)
        print(f"  > Split {i+1}/{splits}: Best AUC {score:.4f}")


    final_model = XGBClassifier(**final_params)
    final_model.fit(X_scaled, y)
    
    return all_preds, np.mean(aucs), final_model, scaler

#  4.BACKTEST 

def run_backtest(data, proba, threshold=0.55, vol_target=0.15, trans_cost=0.0005):
    """
    Backtest with Volatility Targeting.
    If volatility is high, we reduce position size.
    """
    df = data.copy()
    df['proba'] = proba
    df = df.dropna(subset=['proba'])
    
    # 1. Signal Generation
    df['signal'] = 0
    df.loc[df['proba'] > threshold, 'signal'] = 1  # Long
    df.loc[df['proba'] < (1 - threshold), 'signal'] = -1 # Short (optional, remove if Long only)
    
    # 2. Position Sizing (Volatility Targeting)
    # Target 15% annualized vol. If current vol is 30%, we hold 0.5 leverage.
    # Current annualized vol
    curr_vol = df['gk_vol'] * np.sqrt(252)
    df['leverage'] = (vol_target / curr_vol).clip(upper=1.5) # Cap leverage at 1.5x
    
    # Shift signal to t+1 (we trade at Open of next day based on Close of t)
    df['position'] = df['signal'].shift(1) * df['leverage'].shift(1)
    
    # 3. Returns
    # Raw return
    df['strat_ret'] = df['position'] * df['target_ret'] # target_ret is already next day return
    
    # Transaction costs
    turnover = df['position'].diff().abs()
    df['costs'] = turnover * trans_cost
    df['strat_ret_net'] = df['strat_ret'] - df['costs']
    
    # 4. Metrics
    df['equity'] = (1 + df['strat_ret_net']).cumprod()
    
    sharpe = df['strat_ret_net'].mean() / df['strat_ret_net'].std() * np.sqrt(252)
    max_dd = (df['equity'] / df['equity'].cummax() - 1).min()
    
    return {
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "WinRate": (df['strat_ret_net'] > 0).mean(),
        "FinalReturn": df['equity'].iloc[-1] - 1
    }, df['equity']

# --- 5. MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="BTC-USD")
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    print(f"--- AI TRADING AGENT: {args.ticker} ---")
    
    # 1. Load
    raw = load_data(args.ticker)
    
    # 2. Features
    print("Generating advanced features...")
    data, feats = feature_engineering(raw)
    print(f"Total Features: {len(feats)}")

    # 3. Train
    print("Training with Optuna hyperparameter tuning...")
    probas, mean_auc, model, scaler = train_optimized_model(data, feats)
    print(f"\nAverage Out-of-Sample AUC: {mean_auc:.4f}")
    if mean_auc < 0.52:
        print("WARNING: Model has weak predictive power (AUC < 0.52).")

    # 4. Backtest
    print("\nRunning Backtest (with Volatility Targeting)...")
    stats, equity = run_backtest(data, probas, threshold=0.55)
    print(pd.Series(stats))

    # 5. Prediction
    last_row = data.iloc[[-1]][feats]
    last_scaled = scaler.transform(last_row)
    next_prob = model.predict_proba(last_scaled)[:, 1][0]
    
    print(f"\n--- FORECAST for Next Day ---")
    print(f"Probability UP: {next_prob:.2%}")
    
    if next_prob > 0.55:
        print("Action: BUY")
    elif next_prob < 0.45:
        print("Action: SELL/SHORT")
    else:
        print("Action: HOLD / CASH")

if __name__ == "__main__":
    main()
