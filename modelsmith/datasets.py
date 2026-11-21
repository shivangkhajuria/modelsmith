"""A synthetic ad-click dataset for demos and tests.

It looks like a real AdTech CTR table and has two traps an honest modeller
should catch: an ID column, and a post-click column that leaks the label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_ad_clicks(n: int = 20_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 65, n)
    gender = rng.choice(["M", "F"], n, p=[0.62, 0.38])
    city_tier = rng.choice(["tier1", "tier2", "tier3"], n, p=[0.45, 0.35, 0.20])
    device = rng.choice(["android", "ios", "web"], n, p=[0.70, 0.18, 0.12])
    hour = rng.integers(0, 24, n)
    ad_category = rng.choice(["fintech", "gaming", "ecommerce", "travel", "edtech"], n)
    ad_position = rng.choice(["top", "feed", "bottom"], n, p=[0.3, 0.5, 0.2])
    imps = rng.poisson(40, n)
    clicks_7d = rng.binomial(imps, rng.beta(1.5, 40, n))
    session_sec = np.round(rng.gamma(2.0, 60.0, n), 1)
    is_premium = rng.random(n) < 0.08

    hist_ctr = clicks_7d / np.maximum(imps, 1)
    # Mostly non-linear effects and interactions, so tree models and
    # feature engineering pay off over a plain linear baseline.
    logit = (
        -3.6
        + 2.2 * np.tanh(25.0 * hist_ctr)
        + 0.6 * (ad_position == "top")
        - 0.7 * (ad_position == "bottom")
        + 1.3 * ((ad_category == "gaming") & (age < 28))
        + 1.1 * ((ad_category == "fintech") & (city_tier == "tier1") & (age > 30))
        + 0.9 * ((ad_category == "travel") & (hour >= 18))
        + 0.8 * np.exp(-((hour - 21) ** 2) / 6.0)
        - 1.0 * is_premium
        + 1.0 * ((session_sec > 90) & (session_sec < 300))
        + 0.6 * ((device == "ios") & (ad_category == "ecommerce"))
        + rng.normal(0, 0.3, n)
    )
    clicked = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame(
        {
            "user_id": [f"u{100000 + i}" for i in range(n)],
            "age": age,
            "gender": gender,
            "city_tier": city_tier,
            "device": device,
            "hour": hour,
            "ad_category": ad_category,
            "ad_position": ad_position,
            "impressions_7d": imps,
            "clicks_7d": clicks_7d,
            "session_seconds": session_sec,
            "is_premium": is_premium.astype(int),
            # Leak: only non-zero after a click happened.
            "landing_dwell_seconds": np.where(clicked == 1, np.round(rng.gamma(3, 10, n), 1), 0.0),
            "clicked": clicked,
        }
    )
    # Realistic mess: some missing values.
    df.loc[rng.random(n) < 0.04, "age"] = np.nan
    df.loc[rng.random(n) < 0.03, "session_seconds"] = np.nan
    return df


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "examples/ad_clicks.csv"
    make_ad_clicks().to_csv(out, index=False)
    print(f"wrote {out}")
