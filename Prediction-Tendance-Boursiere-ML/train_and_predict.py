
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss

from xgboost import XGBClassifier


#  Utils


def to_series(x, index=None):
    """Assure une série 1D quel que soit l'objet d'entrée."""
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    arr = np.asarray(x).reshape(-1,)
    return pd.Series(arr, index=index)


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(close, window=14):
    diff = close.diff()
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))


def compute_macd(close):
    macd = ema(close, 12) - ema(close, 26)
    signal = ema(macd, 9)
    return macd, signal


def true_range(h, l, c):
    prev = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev).abs()
    tr3 = (l - prev).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def compute_atr(h, l, c, window=14):
    tr = true_range(h, l, c)
    return tr.ewm(alpha=1/window, adjust=False).mean()


def sharpe_ratio(r):
    r = np.asarray(r)
    if r.std() == 0:
        return 0.0
    return r.mean() / r.std() * np.sqrt(252)

#  Data Loading


def load_price(ticker, start="2015-01-01", end=None):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    df = df.rename(columns=str.lower)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    for c in df.columns:
        df[c] = to_series(df[c], df.index)
    return df


def load_context(index_sym="^GSPC", vix_sym="^VIX", start="2015-01-01", end=None):
    ctx = yf.download([index_sym, vix_sym], start=start, end=end, progress=False)["Close"]
    if isinstance(ctx, pd.Series):
        ctx = ctx.to_frame()
    ctx = ctx.rename(columns={index_sym: "index", vix_sym: "vix"}).dropna()
    return ctx


#  Features


def build_dataset(px, horizon=1, ctx=None):
    out = px.copy()

    close = out["close"]
    high = out["high"]
    low = out["low"]

    # Target: direction of future return
    out["ret1"] = np.log(close).diff()
    out["y"] = (np.log(close.shift(-horizon)) - np.log(close) > 0).astype(int)

    # Core features
    out["rsi"] = compute_rsi(close, 14)
    out["macd"], out["macd_signal"] = compute_macd(close)
    out["atr"] = compute_atr(high, low, close)
    out["sma10"] = close.rolling(10).mean()
    out["sma20"] = close.rolling(20).mean()
    out["mom5"] = close.pct_change(5)
    out["vol20"] = out["ret1"].rolling(20).std()

    # Trend slope (small linear fit)
    out["slope20"] = close.rolling(20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0],
        raw=False
    )

    features = [
        "rsi", "macd", "macd_signal", "atr",
        "sma10", "sma20", "mom5", "vol20", "slope20"
    ]

    if ctx is not None:
        ctx = ctx.reindex(out.index).ffill().bfill()

        idx = ctx["index"]
        vix = ctx["vix"]

        idx_ret = np.log(idx).diff()
        rel = out["ret1"] - idx_ret

        out["rel5"] = rel.rolling(5).mean()
        out["rel20"] = rel.rolling(20).mean()
        out["vix"] = vix
        out["vix_chg5"] = vix.pct_change(5)

        features += ["rel5", "rel20", "vix", "vix_chg5"]

    out = out.dropna()
    return out, features


#  Model Training 


def walk_forward_train(data, features, splits=5):
    X = data[features]
    y = data["y"]

    tscv = TimeSeriesSplit(n_splits=splits)

    all_proba = pd.Series(index=X.index, dtype=float)
    aucs, losses = [], []

    def model():
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=500,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            ))
        ])

    for tr, te in tscv.split(X):
        pipe = model()
        pipe.fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        p = np.asarray(p).ravel()
        all_proba[X.iloc[te].index] = p
        aucs.append(roc_auc_score(y.iloc[te], p))
        losses.append(log_loss(y.iloc[te], p))

    final = model()
    final.fit(X, y)

    return all_proba, np.mean(aucs), np.mean(losses), final


#  Backtest


def run_backtest(data, proba, threshold=0.55, horizon=1, alpha=0.0, costs_bps=5.0):
    p = proba.copy()

    # optional smoothing
    if alpha > 0:
        p = p.ewm(alpha=alpha, adjust=False).mean()

    p = p.reindex(data.index).dropna()

    ret = data.loc[p.index, "ret1"].values
    signal = (p.values > threshold).astype(int)

    # holding period = horizon days
    if horizon > 1:
        extended = np.convolve(signal, np.ones(horizon, dtype=int), mode="full")
        signal = (extended[: len(signal)] > 0).astype(int)

    turnover = np.abs(np.diff(np.r_[0, signal]))
    cost = turnover * (costs_bps / 10000)

    strat_ret = signal * ret - cost
    equity = np.exp(np.nancumsum(strat_ret))
    bh = np.exp(np.nancumsum(ret))

    maxdd = (1 - equity / np.maximum.accumulate(equity)).max()

    stats = {
        "threshold": threshold,
        "horizon": horizon,
        "alpha": alpha,
        "sharpe_net": sharpe_ratio(strat_ret),
        "max_drawdown": float(maxdd),
        "turnover": float(turnover.mean())
    }

    return stats, equity, bh



#  Predict next


def predict_next_day(data, features, model):
    last = data.iloc[[-1]]
    p = model.predict_proba(last[features])[:, 1]
    return float(np.asarray(p).ravel()[0])


#  Main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--use_context", action="store_true")
    parser.add_argument("--index", default="^GSPC")
    parser.add_argument("--vix", default="^VIX")
    parser.add_argument("--save_reports", action="store_true")
    args = parser.parse_args()

    print(f"Loading data for {args.ticker}…")
    px = load_price(args.ticker, args.start, args.end)

    ctx = None
    if args.use_context:
        print("Loading market context…")
        ctx = load_context(args.index, args.vix, args.start, args.end)

    print("Building features…")
    data, feats = build_dataset(px, args.horizon, ctx)

    print("Training model (walk-forward)…")
    proba, auc, ll, final_model = walk_forward_train(data, feats, args.splits)
    print(f"AUC: {auc:.3f} | LogLoss: {ll:.3f}")

    print("Backtesting strategy…")
    stats, eq, bh = run_backtest(
        data,
        proba,
        threshold=args.threshold,
        horizon=args.horizon,
        alpha=args.alpha
    )
    print(stats)

    if args.save_reports:
        out = Path("reports")
        out.mkdir(exist_ok=True)
        pd.DataFrame({"proba": proba, "y": data["y"]}).to_csv(out / f"{args.ticker}_proba.csv")
        print("Saved predictions to reports/")

    print("Predicting next movement…")
    p_next = predict_next_day(data, feats, final_model)
    print(f"Last date: {data.index[-1].date()}")
    print(f"Next-day probability of increase: {p_next:.3f}")
    print("Decision:", "LONG" if p_next > args.threshold else "CASH")


if __name__ == "__main__":
    main()

