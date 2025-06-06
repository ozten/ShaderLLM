#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;

void main(){
  vec2 st = gl_FragCoord.xy/u_resolution.xy;
  st.x *= u_resolution.x/u_resolution.y;
  
  // Distance to the center
  vec2 pos = vec2(0.5)-st;
  float r = length(pos)*2.0;
  float a = atan(pos.y,pos.x);
  
  // Drawing with the distance field
  float f = cos(a*6.0);
  gl_FragColor = vec4(vec3(0., 1.-smoothstep(f,f+0.02,r), 0.0) ,1.0);
  
}
