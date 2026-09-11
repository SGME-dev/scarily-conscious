extends Camera3D

@export var distance: float = 0.05
@export var min_distance: float = 0.0001
@export var max_distance: float = 0.5
@export var zoom_speed: float = 0.002

var yaw: float = 0.0
var pitch: float = 0.0
var is_rotating: bool = false

func _ready() -> void:
	# Crucial for microscopic viewing without close-up clipping
	near = 0.0001

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT or event.button_index == MOUSE_BUTTON_RIGHT:
			is_rotating = event.pressed
		elif event.button_index == MOUSE_BUTTON_WHEEL_UP:
			distance = clamp(distance - (distance * 0.15), min_distance, max_distance)
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			distance = clamp(distance + (distance * 0.15), min_distance, max_distance)

	elif event is InputEventMouseMotion and is_rotating:
		yaw -= event.relative.x * 0.005
		pitch -= event.relative.y * 0.005
		pitch = clamp(pitch, -deg_to_rad(85), deg_to_rad(85))

func _process(_delta: float) -> void:
	var target_pos = Vector3.ZERO
	var rot_basis = Basis.from_euler(Vector3(pitch, yaw, 0))
	global_position = target_pos + rot_basis * Vector3(0, 0, distance)
	look_at(target_pos, Vector3.UP)
