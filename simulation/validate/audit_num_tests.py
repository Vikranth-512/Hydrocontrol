"""
Deep audit of NUM-01 (Timestep Sensitivity) and NUM-02 (Long Horizon Stability).
Traces the dynamics step-by-step to identify root causes of:
  1. The abrupt biomass collapse in NUM-02
  2. Anomalous dt=300s divergence in dissolved_nutrient_mass in NUM-01
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from simulation.dynamics import TankDynamicsParams, TankState, step_dynamics, thermal_efficiency, q10_metabolism_factor
from simulation.validate.common import make_initial_state

def audit_long_horizon():
    """Trace the collapse mechanism in the 100k-step run."""
    print("=" * 80)
    print("AUDIT: NUM-02 Long Horizon Stability")
    print("=" * 80)
    
    params = TankDynamicsParams()
    dt = 60.0
    dt_scale = dt / 60.0
    length = 100_000
    actions = [(1.0, 10.0)] * length
    s = make_initial_state(params, dissolved_mass=2.0, biomass=100.0)
    
    # Track key state at critical points
    milestones = []
    prev_biomass = s.algae_biomass
    collapse_start = None
    peak_biomass = 0.0
    peak_step = 0
    
    for t in range(length):
        fr, dur = actions[t]
        
        # Compute key intermediate quantities for diagnosis
        osmotic_stress = (s.dissolved_nutrient_mass / params.osmotic_half_effect) ** 2
        osmotic_factor = 1.0 / (1.0 + osmotic_stress)
        
        L2 = params.light_attenuation_half_mass ** 2
        light_factor = L2 / (L2 + s.algae_biomass ** 2)
        
        thermal_growth = thermal_efficiency(s.water_temp, params)
        thermal_resp = q10_metabolism_factor(s.water_temp, params)
        
        max_reserve = s.algae_biomass * params.internal_capacity
        reserve_ratio = s.internal_reserve / max(max_reserve, 1e-6)
        
        maintenance_cost = params.maintenance_cost * s.algae_biomass * thermal_resp * dt_scale
        actual_maint = min(maintenance_cost, s.internal_reserve)
        maint_deficit = maintenance_cost - actual_maint
        deficit_ratio = maint_deficit / max(maintenance_cost, 1e-6)
        
        # Mortality computation (mirrors step_dynamics)
        damage_penalty = 1.0 + 5.0 * (s.damage_index ** 2)
        osmotic_penalty = 1.0 + np.log1p(osmotic_stress)
        mortality_rate = params.mortality_rate * thermal_resp * osmotic_penalty * damage_penalty
        mortality_amount = s.algae_biomass * (1.0 - np.exp(-mortality_rate * dt_scale))
        
        # Growth computation
        monod = s.dissolved_nutrient_mass / (params.half_saturation_mass + s.dissolved_nutrient_mass)
        reserve_deficit_frac = max(0.0, max_reserve - s.internal_reserve) / max(max_reserve, 1e-6)
        uptake_rate = params.maximum_uptake_rate * thermal_growth * s.health_index * osmotic_factor * reserve_deficit_frac * monod
        uptake_mass = uptake_rate * s.algae_biomass * dt_scale
        
        growth_drive = reserve_ratio * thermal_growth * osmotic_factor
        growth_amount = params.maximum_growth_rate * growth_drive * light_factor * s.algae_biomass * dt_scale
        
        net_growth = growth_amount - mortality_amount
        
        # Track biomass peak
        if s.algae_biomass > peak_biomass:
            peak_biomass = s.algae_biomass
            peak_step = t
        
        # Detect collapse onset
        if collapse_start is None and s.algae_biomass < peak_biomass * 0.5 and peak_biomass > 200:
            collapse_start = t
        
        # Log at key intervals
        if t % 500 == 0 or (collapse_start and t >= collapse_start - 10 and t <= collapse_start + 200 and t % 5 == 0):
            milestones.append({
                'step': t,
                'time_min': t * dt / 60.0,
                'biomass': s.algae_biomass,
                'dissolved': s.dissolved_nutrient_mass,
                'reserve': s.internal_reserve,
                'health': s.health_index,
                'damage': s.damage_index,
                'osmotic_stress': osmotic_stress,
                'osmotic_factor': osmotic_factor,
                'light_factor': light_factor,
                'reserve_ratio': reserve_ratio,
                'growth_amount': growth_amount,
                'mortality_amount': mortality_amount,
                'net_growth': net_growth,
                'mortality_rate': mortality_rate,
                'damage_penalty': damage_penalty,
                'osmotic_penalty': osmotic_penalty,
                'maintenance_cost': maintenance_cost,
                'maint_deficit': maint_deficit,
                'deficit_ratio': deficit_ratio,
                'uptake_mass': uptake_mass,
                'dead_pool': s.dead_biomass_pool,
            })
        
        s_next = step_dynamics(s, fr, dur, dt, params)
        s = s_next
    
    df = pd.DataFrame(milestones)
    
    print(f"\nPeak biomass: {peak_biomass:.2f} at step {peak_step} (t={peak_step*dt/60:.0f} min)")
    if collapse_start:
        print(f"Collapse onset (biomass < 50% peak): step {collapse_start} (t={collapse_start*dt/60:.0f} min)")
    print(f"Final biomass: {s.algae_biomass:.6e}")
    print(f"Final health: {s.health_index:.6e}")
    print(f"Final dissolved: {s.dissolved_nutrient_mass:.4f}")
    
    # Print the critical transition window
    print("\n--- State around biomass peak and collapse ---")
    critical = df[(df['step'] >= peak_step - 1000) & (df['step'] <= peak_step + 3000)]
    if len(critical) > 0:
        cols = ['step', 'time_min', 'biomass', 'dissolved', 'reserve', 'health', 'damage',
                'osmotic_stress', 'light_factor', 'growth_amount', 'mortality_amount', 'net_growth',
                'mortality_rate', 'damage_penalty', 'osmotic_penalty', 'maintenance_cost', 'maint_deficit']
        pd.set_option('display.max_columns', 20)
        pd.set_option('display.width', 300)
        pd.set_option('display.float_format', '{:.6f}'.format)
        print(critical[cols].to_string(index=False))
    
    # Analyze the causal chain
    print("\n--- Causal Analysis ---")
    
    # Phase 1: Growth phase
    growth_phase = df[df['biomass'] > df['biomass'].iloc[0]]
    if len(growth_phase) > 0:
        print(f"Growth phase: steps 0 to ~{growth_phase['step'].iloc[-1]}")
        print(f"  Biomass grew from {df['biomass'].iloc[0]:.1f} to {peak_biomass:.1f}")
    
    # Find the point where dissolved starts spiking
    dissolved_spike = df[df['dissolved'] > 10]
    if len(dissolved_spike) > 0:
        print(f"\nDissolved nutrient spike begins at step {dissolved_spike['step'].iloc[0]}")
        row = dissolved_spike.iloc[0]
        print(f"  At this point:")
        print(f"    Biomass = {row['biomass']:.1f}")
        print(f"    Osmotic stress = {row['osmotic_stress']:.4f}")
        print(f"    Osmotic factor (uptake suppression) = {row['osmotic_factor']:.4f}")
        print(f"    Light factor = {row['light_factor']:.4f}")
        print(f"    Damage = {row['damage']:.4f}")
        print(f"    Health = {row['health']:.4f}")
    
    # Dose rate vs capacity analysis
    dose_per_step = 1.0 * 10.0 / 60.0  # flowrate * duration / 60
    print(f"\n--- Dose vs Capacity Analysis ---")
    print(f"Dose per step: {dose_per_step:.4f} mass units")
    print(f"Dose per minute: {dose_per_step:.4f} (dt=60s => 1 step = 1 min)")
    
    # At peak biomass, what was the uptake capacity?
    peak_rows = df[df['step'] == peak_step]
    if len(peak_rows) > 0:
        r = peak_rows.iloc[0]
        print(f"\nAt peak biomass ({peak_biomass:.0f}):")
        print(f"  Uptake mass/step: {r['uptake_mass']:.6f}")
        print(f"  Mortality/step: {r['mortality_amount']:.4f}")
        print(f"  Growth/step: {r['growth_amount']:.6f}")
        print(f"  Net growth: {r['net_growth']:.6f}")
        print(f"  Dissolved: {r['dissolved']:.4f}")
        print(f"  Light factor: {r['light_factor']:.6f}")
        
    return df


def audit_timestep_sensitivity():
    """Trace why dt=300s diverges in dissolved_nutrient_mass."""
    print("\n" + "=" * 80)
    print("AUDIT: NUM-01 Timestep Sensitivity")
    print("=" * 80)
    
    params = TankDynamicsParams()
    dts = [30.0, 60.0, 120.0, 300.0]
    total_real_minutes = 500.0
    
    results = {}
    for dt in dts:
        length = int(total_real_minutes * 60.0 / dt)
        actions = [(2.0, dt / 60.0)] * length
        s0 = make_initial_state(params, dissolved_mass=2.0, biomass=100.0)
        
        dt_scale = dt / 60.0
        
        # Run simulation tracking per-step fluxes
        s = s0
        rows = []
        for t in range(min(length, 200)):  # First 200 steps for comparison
            fr, dur = actions[t]
            
            dose_mass = fr * dur / 60.0
            
            osmotic = (s.dissolved_nutrient_mass / params.osmotic_half_effect) ** 2
            osmotic_f = 1.0 / (1.0 + osmotic)
            thermal_g = thermal_efficiency(s.water_temp, params)
            thermal_r = q10_metabolism_factor(s.water_temp, params)
            
            dil_rate = (params.background_dilution_rate + params.ec_decay_jitter) * thermal_r
            dilution = s.dissolved_nutrient_mass * (1.0 - np.exp(-dil_rate * dt_scale))
            
            monod = s.dissolved_nutrient_mass / (params.half_saturation_mass + s.dissolved_nutrient_mass)
            max_reserve = s.algae_biomass * params.internal_capacity
            reserve_deficit = max(0.0, max_reserve - s.internal_reserve) / max(max_reserve, 1e-6)
            uptake_rate = params.maximum_uptake_rate * thermal_g * s.health_index * osmotic_f * reserve_deficit * monod
            uptake_mass = uptake_rate * s.algae_biomass * dt_scale
            
            # Key: dose_mass is the same per real-time-minute but dt_scale changes
            # so dose_mass varies per step!
            rows.append({
                'step': t,
                'real_time_min': t * dt / 60.0,
                'dt': dt,
                'dt_scale': dt_scale,
                'dissolved': s.dissolved_nutrient_mass,
                'biomass': s.algae_biomass,
                'reserve': s.internal_reserve,
                'health': s.health_index,
                'dose_mass_per_step': dose_mass,
                'dilution': dilution,
                'uptake_mass': uptake_mass,
                'dil_rate': dil_rate,
            })
            
            s = step_dynamics(s, fr, dur, dt, params)
        
        results[dt] = pd.DataFrame(rows)
    
    # Compare final dissolved values
    print("\n--- Final State Comparison ---")
    for dt in dts:
        df = results[dt]
        # Get states at similar real-time points
        t100 = df[df['real_time_min'].between(99, 101)]
        t500 = df[df['real_time_min'] >= df['real_time_min'].max() - 1]
        
        print(f"\n  dt={int(dt)}s:")
        if len(t100) > 0:
            r = t100.iloc[0]
            print(f"    At ~100 min: dissolved={r['dissolved']:.6f}, biomass={r['biomass']:.4f}")
        if len(t500) > 0:
            r = t500.iloc[0]
            print(f"    Final:       dissolved={r['dissolved']:.6f}, biomass={r['biomass']:.4f}, reserve={r['reserve']:.4f}")
    
    # Analyze dose mass per real-time-minute
    print("\n--- Dose Mass Analysis ---")
    for dt in dts:
        dt_scale = dt / 60.0
        fr, dur = 2.0, dt / 60.0  # This is the action from the test
        dose_mass = fr * dur / 60.0
        dose_per_real_min = dose_mass / (dt / 60.0)
        print(f"  dt={int(dt)}s: dur={dur:.4f}min, dose_mass/step={dose_mass:.6f}, dose/real_min={dose_per_real_min:.6f}")
    
    # Focus on dissolved trajectory divergence
    print("\n--- Dissolved Nutrient Trajectory at key real-times ---")
    for real_t in [0, 5, 10, 20, 50, 100, 200]:
        print(f"  t={real_t} min:", end="")
        for dt in dts:
            df = results[dt]
            match = df[df['real_time_min'].between(real_t - 0.1, real_t + dt/60.0)]
            if len(match) > 0:
                print(f"  dt={int(dt)}s: {match.iloc[0]['dissolved']:.6f}", end="")
        print()
    
    # Check the dilution rate scaling
    print("\n--- Dilution Scaling Check ---")
    for dt in dts:
        dt_scale = dt / 60.0
        thermal_r = q10_metabolism_factor(22.0, params)  # approx water temp
        dil_rate = (params.background_dilution_rate + params.ec_decay_jitter) * thermal_r
        # Exponential decay factor per step
        decay_factor = 1.0 - np.exp(-dil_rate * dt_scale)
        # Effective decay per real minute
        decay_per_min = 1.0 - (1.0 - decay_factor) ** (60.0 / dt)
        print(f"  dt={int(dt)}s: decay_factor/step={decay_factor:.8f}, effective_decay/min={decay_per_min:.8f}")
    
    # Uptake scaling check
    print("\n--- Uptake Scaling Check (first step) ---")
    for dt in dts:
        df = results[dt]
        r = df.iloc[0]
        print(f"  dt={int(dt)}s: uptake_mass/step={r['uptake_mass']:.8f}, uptake/real_min={r['uptake_mass']/(dt/60.0):.8f}")

    return results


if __name__ == "__main__":
    df_lh = audit_long_horizon()
    results_ts = audit_timestep_sensitivity()
