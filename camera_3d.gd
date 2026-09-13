extends Camera3D

@export var mouse_sensitivity: float = 0.003
@export var move_speed: float = 8.0
@export var boost_multiplier: float = 2.5
@export var slow_multiplier: float = 0.3

var pitch: float = 0.0
var yaw: float = 0.0

func _ready() -> void:
	# Capture mouse cursor on load
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED

func _unhandled_input(event: InputEvent) -> void:
	# Toggle mouse lock with ESC
	if event is InputEventKey and event.is_pressed() and event.keycode == KEY_ESCAPE:
		if Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
			Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
		else:
			Input.mouse_mode = Input.MOUSE_MODE_CAPTURED

	# Mouse look rotation
	if event is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		yaw -= event.relative.x * mouse_sensitivity
		pitch -= event.relative.y * mouse_sensitivity
		pitch = clamp(pitch, deg_to_rad(-89.0), deg_to_rad(89.0))
		
		rotation = Vector3(pitch, yaw, 0.0)

func _process(delta: float) -> void:
	if Input.mouse_mode != Input.MOUSE_MODE_CAPTURED:
		return

	var input_dir := Vector3.ZERO

	# WASD Movement
	if Input.is_key_pressed(KEY_W):
		input_dir.z -= 1.0
	if Input.is_key_pressed(KEY_S):
		input_dir.z += 1.0
	if Input.is_key_pressed(KEY_A):
		input_dir.x -= 1.0
	if Input.is_key_pressed(KEY_D):
		input_dir.x += 1.0

	# Vertical Elevation (E / Q or Space / Shift)
	if Input.is_key_pressed(KEY_E) or Input.is_key_pressed(KEY_SPACE):
		input_dir.y += 1.0
	if Input.is_key_pressed(KEY_Q) or Input.is_key_pressed(KEY_CTRL):
		input_dir.y -= 1.0

	input_dir = input_dir.normalized()

	# Speed Modifiers
	var speed := move_speed
	if Input.is_key_pressed(KEY_SHIFT):
		speed *= boost_multiplier
	elif Input.is_key_pressed(KEY_ALT):
		speed *= slow_multiplier

	# Translate relative to camera rotation basis
	var move_vector := global_transform.basis * input_dir
	global_position += move_vector * speed * delta
