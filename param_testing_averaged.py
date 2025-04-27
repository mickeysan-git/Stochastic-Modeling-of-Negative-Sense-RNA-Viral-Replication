import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
import time
from datetime import datetime

# Reaction rates (per hour)
REACTION_RATES = {
    'k_r1': 1.0, 'k_r2': 10.0, 'k_m': 1e2, 'k_p': 5**2, 'k_bind': 1e-4,
    'k_conj': 1e-4, 'k_v': 1e-6, 'd_r1': 1.0, 'd_r2': 1.0, 'd_m': 0.1,
    'd_p': 1e-4, 'd_np': 0.1, 'd_pp': 0.1, 'd_v': 0
}

# Convert to per minute
REACTION_RATES = {k: v / 60 for k, v in REACTION_RATES.items()}

# Initial quantities of molecules
initial_state = {
    'minus_RNA': 0, 'plus_RNA': 0, 'minus_RNP': 3, 'plus_RNP': 0,
    'mRNA': 0, 'Proteins': 1000, 'Virions': 0
}

# Define species order (this order will be used in arrays)
species_order = ['minus_RNA', 'plus_RNA', 'minus_RNP', 'plus_RNP', 'mRNA', 'Proteins', 'Virions']
initial_state_arr = np.array([initial_state[spec] for spec in species_order], dtype=np.float32)

# Stoichiometry matrix (rows: species, columns: reactions)
stoichiometry = np.array([
    [ 0, +1,  0,  0, -1,  0,  0, -1,  0,  0,  0,  0,  0,  0],
    [+1,  0,  0,  0,  0, -1,  0,  0, -1,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0, +1,  0, -1,  0,  0, -1,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0, +1,  0,  0,  0,  0, -1,  0,  0,  0],
    [ 0,  0, +1,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0],
    [ 0,  0,  0, +1, -1, -1, -1,  0,  0,  0,  0,  0, -1,  0],
    [ 0,  0,  0,  0,  0,  0, +1,  0,  0,  0,  0,  0,  0, -1],
], dtype=np.float32)

# Dependency matrix (determines which species affect each reaction's propensity)
dependency_matrix = np.array([
    [0, 0,  0,  0, +1,  0,  0, +1,  0,  0,  0,  0,  0,  0],
    [0, 0,  0,  0,  0, +1,  0,  0, +1,  0,  0,  0,  0,  0],
    [1, 0, +1,  0,  0,  0, +1,  0,  0, +1,  0,  0,  0,  0],
    [0, 1,  0,  0,  0,  0,  0,  0,  0,  0, +1,  0,  0,  0],
    [0, 0,  0, +1,  0,  0,  0,  0,  0,  0,  0, +1,  0,  0],
    [0, 0,  0,  0, +1, +1, +1,  0,  0,  0,  0,  0, +1,  0],
    [0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, +1],
], dtype=np.float32)

# Convert reaction rates to an array (assuming the insertion order is the intended reaction order)
reaction_rate_arr = np.array(list(REACTION_RATES.values()), dtype=np.float32)

# Add a Lysis Threshold -- Cell lyse when certain amount of virions are produced
LYSIS_THRESHOLD = 1e4

@njit
def gillespie_simulation(initial_state, reaction_rates, time_limit, stoichiometry, dependency_matrix):
    current_time = 0.0
    state = initial_state.copy()
    record_interval = 1 # Record every minute
    last_recorded_time = current_time
    history_times = [current_time]
    history_states = [state.copy()]
    step_count = 0  # Track number of steps
    recorded_steps_count = 0  # Track the number of recorded steps
    lysis_time = None
    time_to_first_virion = None

    num_reactions = reaction_rates.shape[0]
    num_species = state.shape[0]

    while current_time < time_limit:
        step_count += 1  # Increment step counter
        
        # Compute propensities for each reaction
        propensities = np.empty(num_reactions, dtype=np.float32)
        for j in range(num_reactions):
            prod = 1.0
            for i in range(num_species):
                if dependency_matrix[i, j] != 0:
                    prod *= state[i] ** dependency_matrix[i, j]
            propensities[j] = reaction_rates[j] * prod
        
        total_propensity = 0.0
        for j in range(num_reactions):
            total_propensity += propensities[j]
        
        if total_propensity == 0.0:
            break
        
        # Time to next reaction
        r1 = np.random.rand()
        dt = -np.log(r1) / total_propensity
        current_time += dt
        
        # Determine which reaction occurs
        r2 = np.random.rand() * total_propensity
        cumulative = 0.0
        reaction_index = 0
        for j in range(num_reactions):
            cumulative += propensities[j]
            if cumulative >= r2:
                reaction_index = j
                break
        
        # Update state according to the chosen reaction's stoichiometry
        for i in range(num_species):
            state[i] += stoichiometry[i, reaction_index]
            if state[i] < 0:
                state[i] = 0.0

        # Check for lysis condition (if virions exceed threshold)
        if state[6] >= LYSIS_THRESHOLD:  # Virions are at index 6
            lysis_time = current_time
            break
            
        
        # Check for time to first virion
        if state[6] > 0 and time_to_first_virion is None:
            time_to_first_virion = current_time
        
        # Record the history only every record_interval minutes
        if current_time - last_recorded_time >= record_interval:
            recorded_steps_count += 1  # Increment recorded steps counter
            history_times.append(current_time)
            history_states.append(state.copy())
            last_recorded_time = current_time
    
    # Ensure final state is recorded if not already
    if history_times[-1] != current_time:
        history_times.append(current_time)
        history_states.append(state.copy())
    
    return history_times, history_states, step_count, recorded_steps_count, lysis_time, time_to_first_virion

def main():
    output_dir = "param_testing_results"
    os.makedirs(output_dir, exist_ok=True)

    total_results = []

    for reaction_rate_key in REACTION_RATES.keys():
        print(f"\n--- Running tests for {reaction_rate_key} ---")

        initial_value = REACTION_RATES[reaction_rate_key]
        testing_range = np.logspace(np.log10(0.01), np.log10(100), num=5)

        for factor in testing_range:
            # print(f"Testing {reaction_rate_key} with factor {factor:.1e}...")

            run_results = []

            for i in range(5):  # 5 iterations for stochastic averaging
                modified_rates = REACTION_RATES.copy()
                modified_rates[reaction_rate_key] = initial_value * factor
                reaction_rate_arr = np.array(list(modified_rates.values()), dtype=np.float32)

                history_times, history_states, total_steps, recorded_steps, lysis_time, time_to_first_virion = gillespie_simulation(
                    initial_state_arr, reaction_rate_arr, time_limit=600,
                    stoichiometry=stoichiometry, dependency_matrix=dependency_matrix
                )

                final_virions = history_states[-1][6]
                sim_time = history_times[-1]
                virion_rate = final_virions / sim_time if sim_time > 0 else 0

                run_results.append({
                    "Simulation Time": sim_time,
                    "Final Virions": final_virions,
                    "Virion Production Rate": virion_rate,
                    "Lysis Time": lysis_time if lysis_time is not None else np.nan,
                    "Time to First Virion": time_to_first_virion if time_to_first_virion is not None else np.nan
                })

            # Average the results
            df_runs = pd.DataFrame(run_results)
            avg_results = df_runs.mean(numeric_only=True)

            total_results.append({
                "Reaction Rate Key": reaction_rate_key,
                "Factor": f"{factor:.1e}",
                "Avg Simulation Time (min)": round(avg_results["Simulation Time"], 3),
                "Avg Final Virions": round(avg_results["Final Virions"], 3),
                "Avg Production Rate (per min)": round(avg_results["Virion Production Rate"], 3),
                "Avg Lysis Time (min)": round(avg_results["Lysis Time"], 3) if not np.isnan(avg_results["Lysis Time"]) else "No Lysis",
                "Avg Time to First Virion (min)": round(avg_results["Time to First Virion"], 3) if not np.isnan(avg_results["Time to First Virion"]) else "No Virions"
            })

    # Save final CSV
    final_df = pd.DataFrame(total_results)
    final_filename = os.path.join(output_dir, f"param_testing_averaged.csv")
    final_df.to_csv(final_filename, index=False)
    print(f"\nAll parameter tests saved to '{final_filename}'")

if __name__ == "__main__":
    main()
