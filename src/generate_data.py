"""
Generate a synthetic smart-grid dataset in the SAME format as the public
SGCC (State Grid Corporation of China) electricity-theft dataset.

Why: the real dataset is ~170 MB and needs a Kaggle login. This script lets
you run the whole pipeline in 1 minute. When you are ready, download the real
data (see README) and the training script will use it automatically.

Format (one row per customer):
    CONS_NO, FLAG, <date_1>, <date_2>, ... <date_N>
    FLAG = 1 -> electricity thief, 0 -> normal customer
Values are daily kWh. Some values are missing (NaN), like the real data.

Theft patterns follow the six "false data injection" attacks used in research
(Jokar et al., 2016): scaling, cut-offs, random scaling, mean replacement, etc.
"""
import argparse
import numpy as np
import pandas as pd


def normal_profile(n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Daily consumption of an honest customer: base load + weekly + yearly cycle + noise."""
    days = np.arange(n_days)
    base = rng.uniform(3, 25)                                   # kWh/day
    weekly = 1 + rng.uniform(0.05, 0.25) * np.sin(2 * np.pi * days / 7 + rng.uniform(0, 6))
    yearly = 1 + rng.uniform(0.1, 0.4) * np.sin(2 * np.pi * days / 365 + rng.uniform(0, 6))
    noise = rng.normal(1, rng.uniform(0.15, 0.45), n_days).clip(0.05, None)
    return base * weekly * yearly * noise


def apply_theft(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply one of six theft attacks to an honest profile."""
    x = x.copy()
    n = len(x)
    attack = rng.integers(1, 7)
    start = rng.integers(0, n // 2)            # theft usually starts at some point, not day 0
    seg = slice(start, n)
    if attack == 1:                             # constant under-reporting
        x[seg] *= rng.uniform(0.6, 0.9)
    elif attack == 2:                           # meter bypassed during some hours -> cut-off
        cutoff = np.quantile(x, rng.uniform(0.6, 0.9))
        x[seg] = np.minimum(x[seg], cutoff)
    elif attack == 3:                           # random scaling every day
        x[seg] *= rng.uniform(0.5, 1.0, n - start)
    elif attack == 4:                           # report the mean * small factor
        x[seg] = x[seg].mean() * rng.uniform(0.6, 0.9) * rng.normal(1, 0.15, n - start).clip(0.3, None)
    elif attack == 5:                           # zero readings on random days
        mask = rng.random(n - start) < rng.uniform(0.1, 0.3)
        x[start:][mask] = 0.0
    else:                                       # mean * random factor per day
        x[seg] = x[seg].mean() * rng.uniform(0.4, 1.0, n - start)
    return x


def main(n_customers: int, n_days: int, theft_ratio: float, out: str, seed: int):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D").strftime("%Y/%m/%d")
    rows, flags = [], []
    for _ in range(n_customers):
        x = normal_profile(n_days, rng)
        is_thief = rng.random() < theft_ratio
        if is_thief:
            x = apply_theft(x, rng)
        # Hard cases: some honest customers have holidays / vacant periods
        if not is_thief and rng.random() < 0.25:
            s = rng.integers(0, n_days - 30)
            x[s:s + rng.integers(7, 30)] *= rng.uniform(0.0, 0.3)
        # Honest behaviour changes: family moves out, new efficient appliances, solar...
        if not is_thief and rng.random() < 0.20:
            x[rng.integers(n_days // 4, n_days):] *= rng.uniform(0.5, 0.85)
        # Sensor glitches: rare huge spikes
        if rng.random() < 0.05:
            x[rng.integers(0, n_days)] *= rng.uniform(10, 40)
        # Missing values, like the real dataset
        x[rng.random(n_days) < rng.uniform(0, 0.05)] = np.nan
        rows.append(np.round(x, 2))
        flags.append(int(is_thief))

    df = pd.DataFrame(rows, columns=dates)
    df.insert(0, "FLAG", flags)
    df.insert(0, "CONS_NO", [f"C{i:06d}" for i in range(n_customers)])
    df.to_csv(out, index=False)
    print(f"Saved {out}: {n_customers} customers x {n_days} days, "
          f"{df.FLAG.mean():.1%} thieves")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--customers", type=int, default=6000)
    p.add_argument("--days", type=int, default=364)   # 52 full weeks
    p.add_argument("--theft-ratio", type=float, default=0.12)
    p.add_argument("--out", default="data/synthetic_sgcc.csv")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.customers, a.days, a.theft_ratio, a.out, a.seed)
