// Logical signed 64-bit fixed-point sums, using portable u32 atomics. Calls on
// a sum may interleave: low-word carries are accumulated into the high word.
// Read only after the dispatch has finished. The caller checks its per-input
// bound before adding; at scale 2^36, +/-2^26 total units leave a factor-two
// margin below the signed overflow boundary.
const WIDE_SCALE:f32=68719476736.;
const WIDE_MAX_UNITS:f32=67108864.;
const WORD_SCALE:f32=4294967296.;
// The caller declares the storage array `wide_accumulators`.
struct WideSum {low:atomic<u32>,high:atomic<u32>};
fn wide_add(sum_index:u32,value:f32) {
    if value==0. {return;}
    let scaled=abs(value)*WIDE_SCALE;
    let high_part=floor(scaled/WORD_SCALE);
    var high=u32(high_part);
    var low=u32(scaled-high_part*WORD_SCALE);
    if value<0. {
        low=(~low)+1u;
        high=(~high)+select(0u,1u,low==0u);
    }
    let previous=atomicAdd(&wide_accumulators[sum_index].low,low);
    let carry=select(0u,1u,previous+low<previous);
    atomicAdd(&wide_accumulators[sum_index].high,high+carry);
}
fn wide_read(sum_index:u32)->f32 {
    var low=atomicLoad(&wide_accumulators[sum_index].low);
    var high=atomicLoad(&wide_accumulators[sum_index].high);
    let negative=(high&0x80000000u)!=0u;
    if negative {
        low=(~low)+1u;
        high=(~high)+select(0u,1u,low==0u);
    }
    let magnitude=(f32(high)*WORD_SCALE+f32(low))/WIDE_SCALE;
    return select(magnitude,-magnitude,negative);
}
