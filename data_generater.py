import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_comfort_dataset(n_samples: int = 10_000,
                             random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic public-transport comfort dataset that follows
    the distributions and assumptions described in the project report.
    """
    rng = np.random.default_rng(random_state)

    # 1. Simple marginal distributions
    travel_time = rng.normal(loc=80, scale=27.18, size=n_samples).clip(10, None)
    transfers = rng.integers(low=0, high=4, size=n_samples)                # 0‒3 inclusive
    walk_distance = rng.normal(loc=550, scale=174.67, size=n_samples).clip(0, None)
    bus_line = rng.choice(['Line 1', 'Line 2', 'Line 3', 'Line 4', 'Line 5'],
                          size=n_samples)
    weekday = rng.choice(['Mon', 'Tue', 'Wed', 'Thr', 'Fri', 'Sat', 'Sun'], size=n_samples)

    # 2. Departure-time-weighted sampling (minutes after midnight)
    minutes = np.arange(24 * 60)
    weights = np.ones_like(minutes, dtype=float)

    peak_mask = ((minutes >= 7 * 60) & (minutes < 9 * 60)) | \
                ((minutes >= 17 * 60) & (minutes < 19 * 60))
    night_mask = minutes < 5 * 60

    weights[peak_mask] *= 5.0          # boost peaks
    weights[night_mask] *= 0.05        # suppress late night
    weights /= weights.sum()

    departure_minutes = rng.choice(minutes, size=n_samples, p=weights)

    # 3. Congestion level depends on departure time
    congestion = np.empty(n_samples, dtype=float)
    for i, m in enumerate(departure_minutes):
        if (7 * 60) <= m < (9 * 60) or (17 * 60) <= m < (19 * 60):
            mean, sd = 0.85, 0.07
        elif m < 5 * 60:
            mean, sd = 0.15, 0.05
        else:
            mean, sd = 0.35, 0.08
        congestion[i] = rng.normal(mean, sd)
    congestion = congestion.clip(0, 1)

    # 4. Human-readable departure-time strings (HH:MM)
    departure_time = [
        (datetime(1900, 1, 1) + timedelta(minutes=int(m))).strftime('%H:%M')
        for m in departure_minutes
    ]

    # 5. Comfort score (higher is better)
    #    - normalise inputs to [0,1] ranges first
    tt_norm = np.clip((travel_time - 20) / (160 - 20), 0, 1)
    walk_norm = np.clip(walk_distance / 1000, 0, 1)        # assume 0-1 km range
    transfers_norm = transfers / 3

    comfort = 100 - (40 * tt_norm + 20 * walk_norm +
                     30 * congestion + 10 * transfers_norm)

    df = pd.DataFrame({
        'travel_time_min': travel_time,
        'transfers': transfers,
        'walk_distance_m': walk_distance,
        'bus_line': bus_line,
        'congestion': congestion,
        'weekday': weekday,
        'departure_time': departure_time,
        'comfort': comfort
    })

    return df

# Generate the dataset and preview it
df = generate_comfort_dataset()

# Save to CSV for download
file_path = "data.csv"
df.to_csv(file_path, index=False)

file_path