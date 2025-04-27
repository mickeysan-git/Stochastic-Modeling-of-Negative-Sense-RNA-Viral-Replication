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

# Plotting function
def plot_static(history, filename):
    times = history[0]
    history_states = np.array(history[1])
    minus_RNAs = history_states[:, 0]
    plus_RNAs = history_states[:, 1]
    minus_RNPs = history_states[:, 2]
    plus_RNPs = history_states[:, 3]
    mRNAs = history_states[:, 4]
    Proteins = history_states[:, 5]
    Virions = history_states[:, 6]
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, minus_RNAs, label='minus-RNA', color='purple')
    plt.plot(times, plus_RNAs, label='plus-RNA', color='green')
    plt.plot(times, minus_RNPs, label='minus-RNP', color='pink')
    plt.plot(times, plus_RNPs, label='plus-RNP', color='teal')
    plt.plot(times, mRNAs, label='mRNA', color='blue')
    plt.plot(times, Proteins, label='Proteins', color='red')
    plt.plot(times, Virions, label='Virions', color='orange')
    
    plt.title('Stochatsic Model of Negative-Semse RNA Viral Replication')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Quantity')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()
    print("Static plot saved as '{}'".format(filename))

def main():
    
    history_times, history_states, total_steps, recorded_steps, lysis_time, time_to_first_virion = gillespie_simulation(
        initial_state_arr, reaction_rate_arr, time_limit=600,
        stoichiometry=stoichiometry, dependency_matrix=dependency_matrix
    )

    # Plot the results
    plot_history = (history_times, history_states)
    plot_static(plot_history, filename="gillespie_final.png")

     # Print final state as a DataFrame
    final_state = history_states[-1]
    final_state_df = pd.DataFrame({
        'Species': species_order,
        'Quantity': final_state
    })
    print("Final quantities after simulation:")
    print(final_state_df)
    print ("\n")
    
    # Extract key metrics
    final_virions = history_states[-1][6]
    sim_time = history_times[-1]
    virion_rate = final_virions / sim_time if sim_time > 0 else 0

    # Print out the metrics
    print(f"Simulation Time (min): {sim_time:.3f}")
    print(f"Final Virions: {final_virions:.3f}")
    print(f"Virion Production Rate (per min): {virion_rate:.3f}")
    print(f"Lysis Time (min): {round(lysis_time, 3) if lysis_time is not None else 'No Lysis'}")
    print(f"Time to First Virion (min): {round(time_to_first_virion, 3) if time_to_first_virion is not None else 'No Virions'}")

if __name__ == "__main__":
    main()