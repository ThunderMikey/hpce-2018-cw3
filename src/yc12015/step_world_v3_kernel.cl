__kernel void kernel_xy(uint32_t x, uint32_t y, uint32_t w,
    float outer, float inner,
    const float *states, const uint32_t *props,
    float *buffer){
    unsigned index=y*w + x;
    
    if((props[index] & Cell_Fixed) || (props[index] & Cell_Insulator)){
      // Do nothing, this cell never changes (e.g. a boundary, or an interior fixed-value heat-source)
      buffer[index]=states[index];
    }else{
      float contrib=inner;
      float acc=inner*states[index];
      
      // Cell above
      if(! (props[index-w] & Cell_Insulator)) {
        contrib += outer;
        acc += outer * states[index-w];
      }
      
      // Cell below
      if(! (props[index+w] & Cell_Insulator)) {
        contrib += outer;
        acc += outer * states[index+w];
      }
      
      // Cell left
      if(! (props[index-1] & Cell_Insulator)) {
        contrib += outer;
        acc += outer * states[index-1];
      }
      
      // Cell right
      if(! (props[index+1] & Cell_Insulator)) {
        contrib += outer;
        acc += outer * states[index+1];
      }
      
      // Scale the accumulate value by the number of places contributing to it
      float res=acc/contrib;
      // Then clamp to the range [0,1]
      res=std::min(1.0f, std::max(0.0f, res));
      buffer[index] = res;
      
    } // end of if(insulator){ ... } else {

}

// vim: ft=c:
