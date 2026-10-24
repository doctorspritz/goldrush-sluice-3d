// Coefficients use staggered MAC indices. Plane element zero is a ready
// header; dispatch ordering, rather than CPU uniform rewrites, selects the
// ordinary position-repair operator before the phase response is assembled.
override VARIABLE_MOBILITY_ALLOWED:bool=true;
fn variable_mobility()->bool {return VARIABLE_MOBILITY_ALLOWED && mobility_u[0]!=0.;}
fn face_mobility(axis:u32,index:u32)->f32 {
    if !variable_mobility() {return 1.;}
    if axis==0u {return mobility_u[index+1u];}
    if axis==1u {return mobility_v[index+1u];}
    return mobility_w[index+1u];
}
