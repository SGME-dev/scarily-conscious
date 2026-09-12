import json
import numpy as np
from caveclient import CAVEclient

client = CAVEclient('flywire_fafb_public')
synapse_table = 'synapses_nt_v1'

# 1. Fetch an active proofread neuron ID
print("Finding an active neuron...")
neurons_df = client.materialize.query_table('proofread_neurons', limit=10)
target_root_id = int(neurons_df['pt_root_id'].iloc[0])
print(f"Querying synapses for single neuron ID: {target_root_id}")

# 2. Query ALL presynaptic outputs with coordinates converted directly to nanometers [1, 1, 1]
synapses = client.materialize.query_table(
    synapse_table,
    filter_equal_dict={'pre_pt_root_id': target_root_id},
    desired_resolution=[1, 1, 1]
)

# Fallback if target has few outputs: query post-synaptic inputs instead
if len(synapses) < 100:
    synapses = client.materialize.query_table(
        synapse_table,
        filter_equal_dict={'post_pt_root_id': target_root_id},
        desired_resolution=[1, 1, 1]
    )

print(f"Retrieved {len(synapses)} connected synapses for this cell.")

# 3. Neurotransmitter probability column mapping
NT_MAP = {
    'ach': 'acetylcholine',
    'gaba': 'gaba',
    'glut': 'glutamate',
    'oct': 'octopamine',
    'ser': 'serotonin',
    'da': 'dopamine'
}

# Find valid coordinate column
pos_col = next((col for col in ['ctr_pt_position', 'pre_pt_position', 'post_pt_position'] if col in synapses.columns), None)

synapse_data = []

if pos_col and not synapses.empty:
    # Filter rows with non-null positions
    valid_synapses = synapses.dropna(subset=[pos_col]).copy()
    
    # Vectorized conversion of coordinates from nanometers to meters (1 nm = 1e-9 m)
    coords_nm = np.vstack(valid_synapses[pos_col].values)
    coords_m = np.round(coords_nm * 1e-9, 6)
    
    # Extract predicted neurotransmitter type per synapse by highest probability column
    nt_cols = [col for col in NT_MAP.keys() if col in valid_synapses.columns]
    if nt_cols:
        types = valid_synapses[nt_cols].idxmax(axis=1).map(NT_MAP).values
    elif 'nt_type' in valid_synapses.columns:
        types = valid_synapses['nt_type'].values
    else:
        types = ['unknown'] * len(valid_synapses)
        
    # Construct output entries matching {"pos": [...], "type": "..."}
    synapse_data = [
        {
            "pos": pos.tolist(),
            "type": str(nt_type)
        }
        for pos, nt_type in zip(coords_m, types)
    ]

# Save to output file
output_path = "fly_synapses.json"
with open(output_path, "w") as f:
    json.dump(synapse_data, f, indent=2)

print(f"Exported {len(synapse_data)} points with neurotransmitter types to {output_path}")