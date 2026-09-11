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

# 2. Query ALL presynaptic outputs for this single neuron
synapses = client.materialize.query_table(
    synapse_table,
    filter_equal_dict={'pre_pt_root_id': target_root_id}
)

# Fallback if target has few outputs: query post-synaptic inputs instead
if len(synapses) < 100:
    synapses = client.materialize.query_table(
        synapse_table,
        filter_equal_dict={'post_pt_root_id': target_root_id}
    )

print(f"Retrieved {len(synapses)} connected synapses for this cell.")

# 3. Raw FlyWire Voxel Resolution to Meters:
# X = 4nm (4e-9m), Y = 4nm (4e-9m), Z = 40nm (40e-9m)
VOXEL_TO_METERS = [4e-9, 4e-9, 40e-9]

synapse_data = []
cols = list(synapses.columns)

for idx, row in synapses.iterrows():
    pos = None

    for col_name in ['ctr_pt_position', 'pre_pt_position', 'post_pt_position']:
        if col_name in cols and row[col_name] is not None:
            val = row[col_name]
            if isinstance(val, (list, np.ndarray, tuple)) and len(val) == 3:
                pos = [
                    float(val[0]) * VOXEL_TO_METERS[0],
                    float(val[1]) * VOXEL_TO_METERS[1],
                    float(val[2]) * VOXEL_TO_METERS[2]
                ]
                break

    if pos is not None:
        synapse_data.append({"pos": pos})

output_path = "fly_synapses.json"
with open(output_path, "w") as f:
    json.dump(synapse_data, f, indent=2)

print(f"Exported {len(synapse_data)} points to {output_path}")