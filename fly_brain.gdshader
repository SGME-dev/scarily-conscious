shader_type spatial;
render_mode unshaded; // Unshaded makes active firing points glow brightly

void fragment() {
    ALBEDO = COLOR.rgb;
}