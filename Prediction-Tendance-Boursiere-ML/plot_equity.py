
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def sharpe_ratio(r):
    r = np.asarray(r, dtype=float)
    if r.std() == 0:
        return 0.0
    return r.mean() / r.std() * np.sqrt(252)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--costs_bps", type=float, default=5.0)
    args = parser.parse_args()

    # Load probabilities saved by train_and_predict.py
    path = f"reports/{args.ticker}_proba.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    proba = df["proba"] if "proba" in df.columns else df["proba_up"]

    # Load price
    px = yf.download(args.ticker, start=args.start, end=args.end, progress=False)["Close"]
    px = px.reindex(proba.index).ffill()
    ret = np.log(px).diff().fillna(0)

    # Optional smoothing
    p = proba.copy()
    if args.alpha > 0:
        p = p.ewm(alpha=args.alpha, adjust=False).mean()

    # Strategy signal
    signal = (p > args.threshold).astype(float).values

    # Transaction costs
    turnover = np.abs(np.diff(np.r_[0, signal]))
    costs = turnover * (args.costs_bps / 10000.0)

    # Strategy returns
    strat_ret = signal * ret.values - costs
    equity = np.exp(np.cumsum(strat_ret))

    # Buy & hold
    bh = np.exp(np.cumsum(ret.values))

    print(f"Sharpe ratio (net): {sharpe_ratio(strat_ret):.2f}")
    print(f"Max Drawdown: {(1 - equity / np.maximum.accumulate(equity)).max():.3f}")
    print(f"Turnover: {turnover.mean():.3f}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(equity, label="Strategy (net costs)")
    plt.plot(bh, label="Buy & Hold")
    plt.title(f"Equity Curve — {args.ticker}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

