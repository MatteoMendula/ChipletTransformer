import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- Configuration ---
FILE_PATH = '/home/matteo/Documents/postDoc/Chetna/chiplet_scheduling/results/llava_ee/llava_ee.csv'
FREQUENCY = 1e9  # Assumed 1 GHz Clock Frequency. Adjust as needed.
MARKERS = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

# Estimated Accuracies (EMNIST) from previous analysis
ACC_EE1 = 94.5
ACC_EE2 = 96.2
ACC_FINAL = 96.5

def load_and_process_data(file_path):
    # Load Data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found at {file_path}. Please check the path.")
        return None

    # Clean column names (remove extra spaces)
    df.columns = [c.strip() for c in df.columns]

    # Ensure required columns exist
    req_cols = ['Layer Number', 'Runtime (Cycles)', 'Activity count-based Energy (nJ)']
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # --- Identify Exit Indices ---
    # We define the cumulative sum of metrics up to specific layers
    # We look for the "Early Exit Head" layers based on the architecture description
    
    # Cumulative Sums
    df['Cum_Cycles'] = df['Runtime (Cycles)'].cumsum()
    df['Cum_Energy_nJ'] = df['Activity count-based Energy (nJ)'].cumsum()

    # Find indices for EE1 (After BLK14), EE2 (After BLK21), and Final
    # We search for string matches in 'Layer Number' column
    # Adjust strings below if CSV layer names differ slightly
    
    try:
        # Finding the index of the specific exit layers
        idx_ee1 = df[df['Layer Number'].astype(str).str.contains('BLK14', case=False) & 
                     df['Layer Number'].astype(str).str.contains('HEAD|EXIT', case=False)].index[0]
        
        idx_ee2 = df[df['Layer Number'].astype(str).str.contains('BLK21', case=False) & 
                     df['Layer Number'].astype(str).str.contains('HEAD|EXIT', case=False)].index[0]
        
        idx_final = df.index[-1] # Assume the last row is the final output
    except IndexError:
        print("Could not find explicit Exit Layer names (e.g., 'BLK14...HEAD'). Using approximate indices based on architecture depth.")
        # Fallback: Approximate locations if names don't match (14/24ths and 21/24ths of the file)
        n_layers = len(df)
        idx_ee1 = int(n_layers * (14/24))
        idx_ee2 = int(n_layers * (21/24))
        idx_final = n_layers - 1

    # Extract Metrics
    # Latency [s] = Cycles / Frequency
    lat_ee1 = df.loc[idx_ee1, 'Cum_Cycles'] / FREQUENCY
    lat_ee2 = df.loc[idx_ee2, 'Cum_Cycles'] / FREQUENCY
    lat_final = df.loc[idx_final, 'Cum_Cycles'] / FREQUENCY

    # Energy [J] = Energy [nJ] * 1e-9
    en_ee1 = df.loc[idx_ee1, 'Cum_Energy_nJ'] * 1e-9
    en_ee2 = df.loc[idx_ee2, 'Cum_Energy_nJ'] * 1e-9
    en_final = df.loc[idx_final, 'Cum_Energy_nJ'] * 1e-9

    results = {
        'latency': np.array([lat_ee1, lat_ee2, lat_final]),
        'energy': np.array([en_ee1, en_ee2, en_final]),
        'accuracy': np.array([ACC_EE1, ACC_EE2, ACC_FINAL])
    }
    return results

def plot_llava(data):
    if data is None: return

    latency = data['latency']
    energy = data['energy']
    accuracy = data['accuracy']

    # Setup Plot
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    ax_qual = axs[0]
    ax_energy = axs[1]

    # Styling constants
    m_size_large = 150
    labelsize_small = 12
    labelsize_large = 14

    # --- Plot 1: Quality (Accuracy) vs Latency ---
    # Plot points: EE1, EE2, Final
    ax_qual.scatter(latency[0], accuracy[0], s=m_size_large, marker=MARKERS[2], color='teal', linewidth=2, label='EE-1')
    ax_qual.scatter(latency[1], accuracy[1], s=m_size_large, marker=MARKERS[-1], color='olivedrab', linewidth=2, label='EE-2')
    ax_qual.scatter(latency[2], accuracy[2], s=m_size_large, marker=MARKERS[4], color='k', linewidth=2, label='Final')

    # Annotations
    ax_qual.text(latency[0], accuracy[0] + 0.2, f'EE-1\n{accuracy[0]}%', fontsize=labelsize_small, color='teal', ha='center')
    ax_qual.text(latency[1], accuracy[1] - 0.4, f'EE-2\n{accuracy[1]}%', fontsize=labelsize_small, color='olivedrab', ha='center')
    ax_qual.text(latency[2], accuracy[2] + 0.2, f'Final\n{accuracy[2]}%', fontsize=labelsize_small, color='black', ha='center')

    # Patches (Visual styling from original snippet)
    # Adjust width/height based on data range
    xlims = (0, latency[2] * 1.2)
    ax_qual.set_xlim(xlims)
    ax_qual.set_ylim(90, 100) # Zoomed in on high accuracy
    
    ax_qual.set_ylabel('Quality Acc [%]', fontsize=labelsize_small)
    ax_qual.set_title('LLaVA Quality vs Latency', fontsize=labelsize_large)
    ax_qual.grid(True, linestyle='--', alpha=0.7)
    ax_qual.legend(loc='lower right')

    # --- Plot 2: Energy vs Latency ---
    ax_energy.set_yscale('log')
    
    # Plot points
    ax_energy.scatter(latency[0], energy[0], s=m_size_large, marker=MARKERS[2], color='teal', linewidth=2)
    ax_energy.scatter(latency[1], energy[1], s=m_size_large, marker=MARKERS[-1], color='olivedrab', linewidth=2)
    ax_energy.scatter(latency[2], energy[2], s=m_size_large, marker=MARKERS[4], color='k', linewidth=2)

    # Annotations
    ax_energy.text(latency[0], energy[0] * 1.2, 'EE-1', fontsize=labelsize_small, color='teal', ha='center')
    ax_energy.text(latency[1], energy[1] * 1.2, 'EE-2', fontsize=labelsize_small, color='olivedrab', ha='center')
    
    # Add colored bands (optional aesthetic matching)
    ax_energy.add_patch(mpatches.Rectangle(xy=(0, 1e-9), width=latency[0], height=1, color='green', alpha=0.05))

    ax_energy.set_xlabel('Latency [s]', fontsize=labelsize_small)
    ax_energy.set_ylabel('Energy [J]', fontsize=labelsize_small)
    ax_energy.set_title('LLaVA Energy vs Latency', fontsize=labelsize_large)
    ax_energy.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    plt.show()

# Run
data = load_and_process_data(FILE_PATH)
plot_llava(data)