uniform sampler2D   u_buffer0;

uniform vec2        u_resolution;

void main (void) {    
    vec3 color = vec3(0.0);
    vec2 st = gl_FragCoord.xy/u_resolution.xy;
    
    // color = vec3(st.x, st.y, 0.5);
    color = vec3(st.x);
    gl_FragColor = vec4(color,1.0);
}