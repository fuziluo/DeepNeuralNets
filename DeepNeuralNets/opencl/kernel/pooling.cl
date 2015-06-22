
#define max(x,y)    ((x) > (y) ? (x) : (y))

float poolingFunc(__global float *preAct, int offset, int rin, int cin) {
    float out = 0;
    switch (poolingType) {
    case AVER:
      for (int i = rin; i < rin + poolHeight; i++) {
        for (int j = cin; j < cin + poolWidth; j++) {
          out += preAct[offset + i * inputFeatureMapsShapeW + j];
        }       
      } 
      out /= poolHeight * poolWidth;
      break;
    case MAX:
      out = preAct[offset + rin * inputFeatureMapsShapeW + cin];
      for (int i = rin; i < rin + poolHeight; i++) {
        for (int j = cin; j < cin + poolWidth; j++) {
          out = max(out, preAct[offset + i * inputFeatureMapsShapeW + j]);
        }       
      }
      break;
    default:
      break;
      
    }
    return out;
}

void poolingBackFunc(__global float *preAct, __global float *prevErrors, float error,
    int offset, int rin, int cin) {
  switch (poolingType) {
  case AVER:
    for (int i = rin; i < rin + poolHeight; i++) {
      for (int j = cin; j < cin + poolWidth; j++) {
        prevErrors[offset + i * inputFeatureMapsShapeW + j] = error / (poolHeight * poolWidth);
      }       
    } 
    break;
  case MAX:
    float act = preAct[offset + rin * inputFeatureMapsShapeW + cin];
    int rMax = rin, cMax = cin;
    for (int i = rin; i < rin + poolHeight; i++) {
      for (int j = cin; j < cin + poolWidth; j++) {
        if (act < preAct[offset + i * inputFeatureMapsShapeW + j]) {
          act = preAct[offset + i * inputFeatureMapsShapeW + j];
          rMax = i;
          cMax = j;
        }
        prevErrors[offset + i * inputFeatureMapsShapeW + j] = 0;
      }       
    }
    prevErrors[offset + rMax * inputFeatureMapsShapeW + cMax] = error;
    break;
  default:
    break;
  }
}


__kernel void forwardPass(__global float *inputFeatureMaps, __global float *outputFeatureMaps)
{
  int lo0 = get_local_id(0), lo1 = get_local_id(1);
  int gb0 = get_global_id(0), gb1 = get_global_id(1);
  if(gb1 < numOfFeatureMaps && gb0 < outputFeatureMapsShapeH * outputFeatureMapsShapeW) {
    for (int i = 0; i < batchSize; i++) {
      int inputBatchOffset = i * numOfFeatureMaps * inputFeatureMapsShapeH * inputFeatureMapsShapeW;
      int outputBatchOffest = i * numOfFeatureMaps * outputFeatureMapsShapeH * outputFeatureMapsShapeW;
      int inputFeatureMapsOffset = inputBatchOffset +  gb1 * inputFeatureMapsShapeH * inputFeatureMapsShapeW;
      int outputFeatureMapsOffset = outputBatchOffest + gb1 * outputFeatureMapsShapeH * outputFeatureMapsShapeW;
      int rout = gb0 / outputFeatureMapsShapeW;
      int cout = gb0 % outputFeatureMapsShapeW;
      int rin = rout * poolHeight;
      int cin = cout * poolWidth;

      outputFeatureMaps[outputFeatureMapsOffset + rout * outputFeatureMapsShapeW + cout] = poolingFunc(inputFeatureMaps, inputFeatureMapsOffset, rin, cin);
    }   
  }
}
__kernel void backprop(__global float *errors, __global float *inputFeatureMaps, __global float *prevErrors)
{
  int lo0 = get_local_id(0), lo1 = get_local_id(1);
  int gb0 = get_global_id(0), gb1 = get_global_id(1);
  if(gb1 < numOfFeatureMaps && gb0 < outputFeatureMapsShapeH * outputFeatureMapsShapeW) {
    for (int i = 0; i < batchSize; i++) {
      int inputBatchOffset = i * numOfFeatureMaps * inputFeatureMapsShapeH * inputFeatureMapsShapeW;
      int outputBatchOffest = i * numOfFeatureMaps * outputFeatureMapsShapeH * outputFeatureMapsShapeW;
      int inputFeatureMapsOffset = inputBatchOffset +  gb1 * inputFeatureMapsShapeH * inputFeatureMapsShapeW;
      int outputFeatureMapsOffset = outputBatchOffest + gb1 * outputFeatureMapsShapeH * outputFeatureMapsShapeW;
      int rout = gb0 / outputFeatureMapsShapeW;
      int cout = gb0 % outputFeatureMapsShapeW;
      int rin = rout * poolHeight;
      int cin = cout * poolWidth;

      poolingBackFunc(inputFeatureMaps, prevErrors, errors[outputFeatureMapsOffset + rout * outputFeatureMapsShapeW + cout], 
          inputFeatureMapsOffset, rin, cin);
    }  
  } 

}
