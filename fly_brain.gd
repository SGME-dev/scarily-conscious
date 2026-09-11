extends Node3D

@onready var multimesh_instance: MultiMeshInstance3D = $MultiMeshInstance3D

# Microscopic scale factor in meters (0.000003 = 3 micrometers)
var point_scale: float = 0.000003

var raw_positions: Array[Vector3] = []
var cloud_center: Vector3 = Vector3.ZERO

# --- Neurotransmitter & Signal Color Palette ---
# Resting background color (Dim Navy)
var resting_color: Color = Color(0.02, 0.05, 0.15) 

# Distinct colors for different brain functions
var signal_palette: Array[Dictionary] = [
	{"name": "Touch / Sensory",  "color": Color(0.0, 0.9, 1.0)},   # Bright Cyan
	{"name": "Dopamine / Reward","color": Color(1.0, 0.85, 0.0)},  # Golden Yellow
	{"name": "Emotion / State",  "color": Color(0.9, 0.1, 1.0)},   # Neon Purple/Magenta
	{"name": "Arousal / Flight", "color": Color(0.1, 1.0, 0.3)},   # Octopamine Emerald Green
	{"name": "Serotonin / Mood", "color": Color(1.0, 0.2, 0.4)}    # Coral Red/Pink
]

# Per-synapse functional assignments and activity tracking
var synapse_signal_indices: Array[int] = []
var firing_intensities: Array[float] = []

# Signal wave simulation variables
var wave_position: float = 0.0
var wave_speed: float = 0.00008
var max_dist: float = 0.0001
var current_wave_signal_type: int = 0

func _ready() -> void:
	print_signal_legend()
	load_and_display_synapses("res://python/fly_synapses.json")

func print_signal_legend() -> void:
	print("--- NEUROTRANSMITTER SIGNAL LEGEND ---")
	print("Cyan   = Touch / Sensory Feedback")
	print("Gold   = Dopamine (Reward & Movement)")
	print("Purple = Emotion & Internal States")
	print("Green  = Octopamine (Fight / Flight / Arousal)")
	print("Pink   = Serotonin (Mood & Sleep Modulation)")
	print("---------------------------------------")

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

	# Calculate cloud center
	cloud_center = Vector3.ZERO
	for s in synapses:
		var p = s["pos"]
		var pos_vec = Vector3(p[0], p[1], p[2])
		raw_positions.append(pos_vec)
		cloud_center += pos_vec
		
		firing_intensities.append(0.0)

	if raw_positions.size() > 0:
		cloud_center /= raw_positions.size()

	# Assign signal categories to synapses based on spatial clusters and branch depth
	max_dist = 0.00001
	for pos in raw_positions:
		var dist = (pos - cloud_center).length()
		if dist > max_dist:
			max_dist = dist

	for i in range(raw_positions.size()):
		var dist = (raw_positions[i] - cloud_center).length()
		# Map distance relative to center into one of the signal categories
		var cluster_id = int((dist / max_dist) * signal_palette.size() + (i % 2)) % signal_palette.size()
		synapse_signal_indices.append(cluster_id)

	# Prepare MultiMesh for per-instance rendering
	var multimesh: MultiMesh = multimesh_instance.multimesh
	multimesh.use_colors = true
	multimesh.instance_count = raw_positions.size()

	update_point_transforms()

func update_point_transforms() -> void:
	var multimesh: MultiMesh = multimesh_instance.multimesh
	if multimesh.instance_count != raw_positions.size():
		multimesh.instance_count = raw_positions.size()

	# Create micro-scale matrix
	var scale_basis = Basis().scaled(Vector3.ONE * point_scale)

	for i in range(raw_positions.size()):
		var pos = raw_positions[i] - cloud_center
		var xform = Transform3D(scale_basis, pos)
		multimesh.set_instance_transform(i, xform)

func _process(delta: float) -> void:
	if raw_positions.is_empty():
		return

	var multimesh: MultiMesh = multimesh_instance.multimesh

	# 1. Advance electrical signal wave front down branches
	wave_position += wave_speed * delta
	if wave_position > max_dist * 1.3:
		wave_position = -max_dist * 0.2 # Loop back to start
		current_wave_signal_type = (current_wave_signal_type + 1) % signal_palette.size() # Switch wave signal type

	# 2. Trigger random individual neurotransmitter sparks
	if randf() < 0.35:
		var random_idx = randi() % raw_positions.size()
		firing_intensities[random_idx] = 1.0

	# 3. Calculate per-instance signal colors
	for i in range(raw_positions.size()):
		var dist = (raw_positions[i] - cloud_center).length()

		# Distance from current wave pulse
		var wave_dist = abs(dist - wave_position)
		var wave_pulse = max(0.0, 1.0 - (wave_dist / 0.000012))

		# Decay individual spark over time
		firing_intensities[i] = max(0.0, firing_intensities[i] - delta * 3.5)

		# Determine active signal color based on synapse type or passing wave
		var synapse_type_id = synapse_signal_indices[i]
		var synapse_target_color: Color = signal_palette[synapse_type_id]["color"]
		var wave_target_color: Color = signal_palette[current_wave_signal_type]["color"]

		# Combine wave activity and individual neurotransmitter spark
		var spark_activity = firing_intensities[i]
		var total_activity = clamp(spark_activity + wave_pulse, 0.0, 1.0)

		# Blend resting blue into specific functional firing colors
		var active_color = synapse_target_color.lerp(wave_target_color, wave_pulse)
		var final_color = resting_color.lerp(active_color, total_activity)

		# Brighten during peak firing emission
		if total_activity > 0.1:
			final_color *= (1.0 + total_activity * 2.5)

		multimesh.set_instance_color(i, final_color)

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed:
		# Press '+' or '=' to adjust size live
		if event.keycode == KEY_EQUAL or event.keycode == KEY_KP_ADD:
			point_scale *= 1.3
			update_point_transforms()
			print("Current Synapse Size: ", point_scale)

		# Press '-' to adjust size live
		elif event.keycode == KEY_MINUS or event.keycode == KEY_KP_SUBTRACT:
			point_scale *= 0.7
			update_point_transforms()
			print("Current Synapse Size: ", point_scale)
