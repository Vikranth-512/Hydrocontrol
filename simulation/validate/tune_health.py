import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from simulation.dynamics import TankDynamicsParams, step_dynamics
from simulation.validate.common import make_initial_state

def run_scenario(damage_rate, repair_rate, length=15000, dt=60.0):
    params = TankDynamicsParams()
    params.damage_rate = damage_rate
    params.repair_rate = repair_rate
    
    actions = [(1.0, 10.0)] * length
    s = make_initial_state(params, dissolved_mass=2.0, biomass=100.0)
    
    hist = []
    for t in range(length):
        fr, dur = actions[t]
        
        hist.append({
            'time_min': t * dt / 60.0,
            'biomass': s.algae_biomass,
            'dissolved': s.dissolved_nutrient_mass,
            'health': s.health_index,
            'damage': s.damage_index,
        })
            
        s = step_dynamics(s, fr, dur, dt, params)
        
    return pd.DataFrame(hist)

candidates = [
    (0.001, 0.001),    # Original
    (0.0001, 0.005),   # 10x smaller damage, 5x larger repair
    (0.00005, 0.01),   # 20x smaller damage, 10x larger repair
    (0.00002, 0.01),   # 50x smaller damage, 10x larger repair
    (0.00001, 0.01),   # 100x smaller damage, 10x larger repair
]

out_dir = Path("validation_report/tuning")
out_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(len(candidates), 2, figsize=(12, 4 * len(candidates)))

for i, (dr, rr) in enumerate(candidates):
    print(f"Testing damage={dr}, repair={rr}...")
    df = run_scenario(dr, rr, length=30000)
    
    ax1 = axes[i, 0]
    ax1.plot(df['time_min'], df['biomass'], label='Biomass')
    ax1.plot(df['time_min'], df['dissolved'], label='Dissolved')
    ax1.set_title(f"dr={dr}, rr={rr} - Mass")
    ax1.legend()
    
    ax2 = axes[i, 1]
    ax2.plot(df['time_min'], df['health'], label='Health', color='green')
    ax2.set_title(f"dr={dr}, rr={rr} - Health")
    ax2.legend()
    
    # Calculate transition region duration (time spent between 0.9 and 0.1 health)
    transition = df[(df['health'] < 0.99) & (df['health'] > 0.01)]
    if len(transition) > 0:
        duration_min = transition['time_min'].iloc[-1] - transition['time_min'].iloc[0]
        print(f"  Transition region (0.99 to 0.01) duration: {duration_min} mins")
    else:
        print(f"  No clear transition region found.")

plt.tight_layout()
plt.savefig(out_dir / "health_tuning.png", dpi=150)
print(f"Saved to {out_dir / 'health_tuning.png'}")
