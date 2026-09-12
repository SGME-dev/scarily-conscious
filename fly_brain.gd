extends Node3D

@onready var multimesh_instance: MultiMeshInstance3D = $MultiMeshInstance3D

# Microscopic scale factor in meters (0.000003 = 3 micrometers)
var point_scale: float = 0.000003

var raw_positions: Array[Vector3] = []
var cloud_center: Vector3 = Vector3.ZERO

# --- Noise Control ---
# Toggle for spontaneous random synaptic noise
var enable_random_noise: bool = true 

# --- Biological Neurotransmitter Palette ---
var signal_palette: Array[Dictionary] = [
	{"name": "Acetylcholine (ACh - Excitatory)", "color": Color(0.0, 0.8, 1.0)},   # Cyan
	{"name": "GABA (Inhibitory)",                "color": Color(1.0, 0.2, 0.2)},   # Red
	{"name": "Glutamate (Glu - Mixed)",          "color": Color(0.2, 0.9, 0.3)},   # Green
	{"name": "Dopamine (DA - Reward/Motion)",    "color": Color(1.0, 0.8, 0.0)},   # Gold
	{"name": "Serotonin (5-HT - Mood/Sleep)",    "color": Color(0.9, 0.1, 0.9)},   # Magenta
	{"name": "Octopamine (OA - Arousal/Flight)", "color": Color(1.0, 0.5, 0.0)},   # Orange
	{"name": "Unknown / Unclassified",           "color": Color(0.4, 0.4, 0.5)}    # Grey
]

# Fast string-to-palette-index lookup table
var nt_lookup: Dictionary = {
	"acetylcholine": 0, "ach": 0,
	"gaba": 1,
	"glutamate": 2,     "glu": 2,
	"dopamine": 3,      "da": 3,
	"serotonin": 4,     "5ht": 4,
	"octopamine": 5,    "oa": 5
}

# Per-synapse signal indices and activity tracking
var synapse_signal_indices: Array[int] = []
var firing_intensities: Array[float] = []

# Directional propagation along physical neuron axis
var synapse_axis_depths: Array[float] = []
var primary_axis: Vector3 = Vector3.FORWARD
var min_depth: float = 0.0
var max_depth: float = 1.0

var action_potential_pos: float = 0.0
var pulse_speed: float = 0.35 # Progress per second along neuron length

func _ready() -> void:
	print_signal_legend()
	load_and_display_synapses("res://python/fly_synapses_raw.json")

func print_signal_legend() -> void:
	print("--- BIOLOGICAL NEUROTRANSMITTER MAP LEGEND ---")
	for entry in signal_palette:
		print(entry["name"])
	print("Press 'N' or 'R' to toggle Spontaneous Random Noise ON/OFF")
	print("-----------------------------------------------")

func load_and_display_synapses(path: String) -> void:
	if not FileAccess.file_exists(path):
		push_error("Synapse file not found: " + path)
		return

	var file = FileAccess.open(path, FileAccess.READ)
	var json_string = file.get_as_text()
	file.close()

	var json = JSON.new()
	if json.parse(json_string) != OK:
		push_error("JSON parse error: " + json.get_error_message())
		return

	var synapses: Array = json.data
	raw_positions.clear()
	firing_intensities.clear()
	synapse_signal_indices.clear()
	synapse_axis_depths.clear()

	# 1. Parse positions and map biological neurotransmitter types
	cloud_center = Vector3.ZERO
	for s in synapses:
		var p = s["pos"]
		var pos_vec = Vector3(p[0], p[1], p[2])
		raw_positions.append(pos_vec)
		cloud_center += pos_vec
		firing_intensities.append(0.0)

		# Read 'type' or 'nt' attribute from JSON
		var nt_raw: String = str(s.get("type", s.get("nt", "unknown"))).to_lower().strip_edges()
		var nt_index: int = nt_lookup.get(nt_raw, 6) # Default to Grey if unmapped
		synapse_signal_indices.append(nt_index)

	if raw_positions.size() > 0:
		cloud_center /= raw_positions.size()

	# 2. Determine primary physical growth axis of the neuron
	var furthest_point = raw_positions[0]
	var max_d = 0.0
	for pos in raw_positions:
		var d = (pos - cloud_center).length_squared()
		if d > max_d:
			max_d = d
			furthest_point = pos

	primary_axis = (furthest_point - cloud_center).normalized()

	# 3. Project each synapse onto the primary axis for directional flow
	min_depth = 999999.0
	max_depth = -999999.0

	for i in range(raw_positions.size()):
		var local_pos = raw_positions[i] - cloud_center
		var depth = local_pos.dot(primary_axis)
		synapse_axis_depths.append(depth)

		if depth < min_depth: min_depth = depth
		if depth > max_depth: max_depth = depth

	# 4. Setup MultiMesh GPU instancing
	var multimesh: MultiMesh = multimesh_instance.multimesh
	multimesh.use_colors = true
	multimesh.instance_count = raw_positions.size()

	update_point_transforms()

func update_point_transforms() -> void:
	var multimesh: MultiMesh = multimesh_instance.multimesh
	if multimesh.instance_count != raw_positions.size():
		multimesh.instance_count = raw_positions.size()

	var scale_basis = Basis().scaled(Vector3.ONE * point_scale)

	for i in range(raw_positions.size()):
		var pos = raw_positions[i] - cloud_center
		var xform = Transform3D(scale_basis, pos)
		multimesh.set_instance_transform(i, xform)

func _process(delta: float) -> void:
	if raw_positions.is_empty():
		return

	var multimesh: MultiMesh = multimesh_instance.multimesh
	var total_length = max_depth - min_depth

	# 1. Advance directional electrical pulse down the primary axis
	action_potential_pos += pulse_speed * delta
	if action_potential_pos > 1.25:
		action_potential_pos = -0.15 # Reset to root

	var current_wave_depth = min_depth + (action_potential_pos * total_length)
	var wave_width = total_length * 0.08 # Front width of pulse

	# 2. Trigger random neurotransmitter releases ONLY if random noise is toggled ON
	if enable_random_noise and randf() < 0.35:
		var random_idx = randi() % raw_positions.size()
		firing_intensities[random_idx] = 1.0

	# 3. Update biological colors frame-by-frame
	for i in range(raw_positions.size()):
		var depth = synapse_axis_depths[i]

		# Pulse strength based on distance to advancing wave front
		var dist_to_wave = abs(depth - current_wave_depth)
		var wave_pulse = max(0.0, 1.0 - (dist_to_wave / wave_width))

		# Spark cooldown
		firing_intensities[i] = max(0.0, firing_intensities[i] - delta * 3.5)

		# Get biological color mapped from JSON
		var nt_index = synapse_signal_indices[i]
		var base_color: Color = signal_palette[nt_index]["color"]

		var total_activity = clamp(firing_intensities[i] + wave_pulse, 0.0, 1.0)

		# Resting state shows dim neurotransmitter color (18% brightness)
		var resting_state = base_color * 0.18
		var active_state = base_color * (1.0 + total_activity * 2.5)

		var final_color = resting_state.lerp(active_state, total_activity)
		multimesh.set_instance_color(i, final_color)

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed:
		# Press 'N' or 'R' to switch between Random Noise and Pure Deterministic Wave
		if event.keycode == KEY_N or event.keycode == KEY_R:
			enable_random_noise = !enable_random_noise
			print("Spontaneous Random Noise: ", "ENABLED" if enable_random_noise else "DISABLED (Pure Wave)")

		# Synapse size controls
		elif event.keycode == KEY_EQUAL or event.keycode == KEY_KP_ADD:
			point_scale *= 1.3
			update_point_transforms()
			print("Current Synapse Size: ", point_scale)
		elif event.keycode == KEY_MINUS or event.keycode == KEY_KP_SUBTRACT:
			point_scale *= 0.7
			update_point_transforms()
			print("Current Synapse Size: ", point_scale)
