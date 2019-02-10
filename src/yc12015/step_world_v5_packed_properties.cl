enum cell_flags_t{
  Cell_Fixed    = 0x1,
  Cell_Insulator= 0x2
};

// packed properties definition
// this:  1-0
// above: 3-2
// below: 5-4
// left:  7-6
// right: 9-8

__kernel void kernel_xy(
    float inner,
    float outer, 
    __global const uint *props,
    __global const float *states,
    __global float *buffer
    ){

  uint x = get_global_id(0);
  uint y = get_global_id(1);
  uint w = get_global_size(0);

  unsigned index=y*w + x;
  
  if((props[index] & Cell_Fixed) || (props[index] & Cell_Insulator)){
    // Do nothing, this cell never changes (e.g. a boundary, or an interior fixed-value heat-source)
    buffer[index]=states[index];
  }else{
    float contrib=inner;
    float acc=inner*states[index];
    
    // Cell above
    if(! (props[index] & (Cell_Insulator << 2))) {
      contrib += outer;
      acc += outer * states[index-w];
    }
    
    // Cell below
    if(! (props[index] & (Cell_Insulator << 4))) {
      contrib += outer;
      acc += outer * states[index+w];
    }
    
    // Cell left
    if(! (props[index] & (Cell_Insulator << 6))) {
      contrib += outer;
      acc += outer * states[index-1];
    }
    
    // Cell right
    if(! (props[index] & (Cell_Insulator << 8))) {
      contrib += outer;
      acc += outer * states[index+1];
    }
    
    // Scale the accumulate value by the number of places contributing to it
    float res=acc/contrib;
    // Then clamp to the range [0,1]
    res=min(1.0f, max(0.0f, res));
    buffer[index] = res;
    
  } // end of if(insulator){ ... } else {

}

// vim: ft=c:
