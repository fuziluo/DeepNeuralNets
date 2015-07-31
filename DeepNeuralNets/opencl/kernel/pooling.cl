
#define min(x,y)    ((x) < (y) ? (x) : (y))
#define max(x,y)    ((x) > (y) ? (x) : (y))
float activationFunction(float input) {
  float output = input;
  switch (activationType) {
  case NONE:
    output = input;
    break;
  case RELU:
    output = max(0, input);
    // output = input > 0 ? input : 0.01 * input;
    break;
  case SIGMOID:
    output = (1/(1 + exp(-input)));
    break;
  case TANH:
    output = 2/(1 + exp(-input)) - 1;
    break;
  default:
    break;
  }
  return output;
}
float derivative(float input) {
  float output = 1;
  switch (prevActivationType) {
  case NONE:
    output = 1;
    break;
  case RELU:
    output = input > 0 ? 1 : 0;
    // output = input > 0 ? 1 : 0.01;
    break;
  case SIGMOID:
    output = input * (1 - input);
    break;
  case TANH:
    output = (input + 1) * (1 - input) / 2;
    break;
  default:
    break;
  }
  return output;
}
/*void atomic_add_global(volatile global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
 
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}*/

#ifdef padding
float poolingFunc(__global float *preAct, int offset, int rin, int cin) {
  float out = 0;
  int cnt = (min(rin + poolHeight, inputFeatureMapsShapeH) - max(rin, 0)) *
          (min(cin + poolWidth, inputFeatureMapsShapeW) - max(cin, 0));
  switch (poolingType) {
  case AVER:
    for (int i = rin; i < rin + poolHeight && i < inputFeatureMapsShapeH; i++) {
      for (int j = cin; j < cin + poolWidth && j < inputFeatureMapsShapeW; j++) {
        out = select(out, out + preAct[offset + i * inputFeatureMapsShapeW + j], i >= 0 && j >= 0);
        // out += preAct[offset + i * inputFeatureMapsShapeW + j];
        // cnt++;
      }       
    } 
    out /= cnt;
    // out /= poolHeight * poolWidth;
    break;
  case MAX:
    out = -FLT_MAX;
    for (int i = rin; i < rin + poolHeight && i < inputFeatureMapsShapeH; i++) {
      for (int j = cin; j < cin + poolWidth && j < inputFeatureMapsShapeW; j++) {
        out = select(out, max(out, preAct[offset + i * inputFeatureMapsShapeW + j]), i >= 0 && j >= 0);
        // out = max(out, preAct[offset + i * inputFeatureMapsShapeW + j]);
      }       
    }
    break;
  default:
    break;
    
  }
  return activationFunction(out);
}


void poolingBackFunc(__global float *errors, int rin, int cin, int inputFeatureMapsOffset, int outputFeatureMapsOffset,
                    __global float *prevError, __global float *inputFeatureMaps) {
  float der = derivative(inputFeatureMaps[inputFeatureMapsOffset + rin * inputFeatureMapsShapeW + cin]);
  // int rout = rin / stride, cout = cin / stride;
  // int r = rout * stride, c = cout * stride;
  int r1 = max(-(poolHeight - 1) / 2, rin - poolHeight + 1);
  int r = ((r1 + (poolHeight - 1) / 2) % stride) ? (r1 + stride - (r1 + (poolHeight - 1) / 2) % stride) : r1;
  int c1 = max(-(poolWidth - 1) / 2, cin - poolWidth + 1);
  int c = ((c1 + (poolWidth - 1) / 2) % stride) ? (c1 + stride - (c1 + (poolWidth - 1) / 2) % stride) : c1;
  switch (poolingType) {
  case AVER:
  {
    for (int i = r; i <= rin && i < inputFeatureMapsShapeH - (poolHeight - 1) / 2; i += stride) {
      for (int j = c; j <= cin && j < inputFeatureMapsShapeW - (poolWidth - 1) / 2; j += stride) {
        int cnt = (min(i + poolHeight, inputFeatureMapsShapeH) - max(i, 0)) * (min(j + poolWidth, inputFeatureMapsShapeW) - max(j, 0));
        int rout = (i + (poolHeight - 1) / 2) / stride;
        int cout = (j + (poolWidth - 1) / 2) / stride;
        *prevError += errors[outputFeatureMapsOffset + rout * outputFeatureMapsShapeW + cout] / cnt * der;
      }
    }
    break;
  } 
  case MAX:
  { 
    for (int i = r; i <= rin && i < inputFeatureMapsShapeH - (poolHeight - 1) / 2; i += stride) {
      for (int j = c; j <= cin && j < inputFeatureMapsShapeW - (poolWidth - 1) / 2; j += stride) {
        // float err = errors[outputFeatureMapsOffset + i / stride * outputFeatureMapsShapeW + j / stride];
        int rmax = 0, cmax = 0;
        float max_in = -FLT_MAX;
        for (int m = i; m < i + poolHeight && m < inputFeatureMapsShapeH; m++) {
          for (int n = j; n < j + poolWidth && n < inputFeatureMapsShapeW; n++) {
            bool cond = m >= 0 && n >= 0 && max_in < inputFeatureMaps[inputFeatureMapsOffset + m * inputFeatureMapsShapeW + n];
            if (cond) {
              max_in = inputFeatureMaps[inputFeatureMapsOffset + m * inputFeatureMapsShapeW + n];
              rmax = m;
              cmax = n;
            }
          }
        }
        int rout = (i + (poolHeight - 1) / 2) / stride;
        int cout = (j + (poolWidth - 1) / 2) / stride;
        if (rmax == rin && cmax == cin)
          *prevError += errors[outputFeatureMapsOffset + rout * outputFeatureMapsShapeW + cout] * der;
      }
    }       
    break;
  }
  default:
    break;
  }
} 
#else
float poolingFunc(__global float *preAct, int offset, int rin, int cin) {
  float out = 0;
  int cnt = poolHeight * poolWidth;
  switch (poolingType) {
  case AVER:
    for (int i = rin; i < rin + poolHeight && rin + poolHeight <= inputFeatureMapsShapeH; i++) {
      for (int j = cin; j < cin + poolWidth && cin + poolWidth <= inputFeatureMapsShapeW; j++) {
        out += preAct[offset + i * inputFeatureMapsShapeW + j];
      }       
    } 
    out /= cnt;
    break;
  case MAX:
    out = preAct[offset + rin * inputFeatureMapsShapeW + cin];
    for (int i = rin; i < rin + poolHeight && rin + poolHeight <= inputFeatureMapsShapeH; i++) {
      for (int j = cin; j < cin + poolWidth && cin + poolWidth <= inputFeatureMapsShapeW; j++) {
        out = max(out, preAct[offset + i * inputFeatureMapsShapeW + j]);
      }       
    }
    break;
  default:
    break;
    
  }
  return activationFunction(out);
}

void poolingBackFunc(__global float *errors, int rin, int cin, int inputFeatureMapsOffset, int outputFeatureMapsOffset,
                    __global float *prevError, __global float *inputFeatureMaps) {
  float der = derivative(inputFeatureMaps[inputFeatureMapsOffset + rin * inputFeatureMapsShapeW + cin]);
  // int rout = rin / stride, cout = cin / stride;
  // int r = rout * stride, c = cout * stride;
  int r = (max(0, rin - poolHeight + 1) % stride) ? (max(0, rin - poolHeight + 1) + stride - max(0, rin - poolHeight + 1) % stride) : max(0, rin - poolHeight + 1);
  int c = (max(0, cin - poolWidth + 1) % stride) ? (max(0, cin - poolWidth + 1) + stride - max(0, cin - poolWidth + 1) % stride) : max(0, cin - poolWidth + 1);
  switch (poolingType) {
  case AVER:
  {
    for (int i = r; i <= rin; i += stride) {
      for (int j = c; j <= cin; j += stride) {
        int cnt = min(poolHeight, inputFeatureMapsShapeH - i) * min(poolWidth, inputFeatureMapsShapeW - j);
        *prevError += errors[outputFeatureMapsOffset + i / stride * outputFeatureMapsShapeW + j / stride] / cnt * der;
      }
    }
    break;
  } 
  case MAX:
  { 
    for (int i = r; i <= rin; i += stride) {
      for (int j = c; j <= cin; j += stride) {
        // float err = errors[outputFeatureMapsOffset + i / stride * outputFeatureMapsShapeW + j / stride];
        int rmax = i, cmax = j;
        for (int m = i; m < i + poolHeight && m < inputFeatureMapsShapeH; m++) {
          for (int n = j; n < j + poolWidth && n < inputFeatureMapsShapeW; n++) {
            if (inputFeatureMaps[inputFeatureMapsOffset + rmax * inputFeatureMapsShapeW + cmax] < inputFeatureMaps[inputFeatureMapsOffset + m * inputFeatureMapsShapeW + n]) {
              rmax = m;
              cmax = n;
            }
          }
        }
        if (rmax == rin && cmax == cin)
          *prevError += errors[outputFeatureMapsOffset + i / stride * outputFeatureMapsShapeW + j / stride] * der;
      }
    }       
    break;
  }
  default:
    break;
  }
} 

#endif

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
      #ifdef padding
      int rin = rout * stride - (poolHeight - 1) / 2;
      int cin = cout * stride - (poolWidth - 1) / 2;
      #else
      int rin = rout * stride;
      int cin = cout * stride;
      #endif

      outputFeatureMaps[outputFeatureMapsOffset + rout * outputFeatureMapsShapeW + cout] = poolingFunc(inputFeatureMaps, inputFeatureMapsOffset, rin, cin);
    }   
  }
}

__kernel void backprop(__global float *errors, __global float *inputFeatureMaps, __global float *prevErrors)
{
  int lo0 = get_local_id(0), lo1 = get_local_id(1);
  int gb0 = get_global_id(0), gb1 = get_global_id(1);
  if(gb1 < numOfFeatureMaps && gb0 < inputFeatureMapsShapeH * inputFeatureMapsShapeW) {
    for (int i = 0; i < batchSize; i++) {
      int inputBatchOffset = i * numOfFeatureMaps * inputFeatureMapsShapeH * inputFeatureMapsShapeW;
      int outputBatchOffest = i * numOfFeatureMaps * outputFeatureMapsShapeH * outputFeatureMapsShapeW;
      int inputFeatureMapsOffset = inputBatchOffset +  gb1 * inputFeatureMapsShapeH * inputFeatureMapsShapeW;
      int outputFeatureMapsOffset = outputBatchOffest + gb1 * outputFeatureMapsShapeH * outputFeatureMapsShapeW;
      int rin = gb0 / inputFeatureMapsShapeW;
      int cin = gb0 % inputFeatureMapsShapeW;
      // int rout = rin / stride;
      // int cout = cin / stride;

      // poolingBackFunc(inputFeatureMaps, prevErrors, errors[outputFeatureMapsOffset + rout * outputFeatureMapsShapeW + cout], 
      //     inputFeatureMapsOffset, rin, cin);
      poolingBackFunc(errors, rin, cin, inputFeatureMapsOffset, outputFeatureMapsOffset,
                      &prevErrors[inputFeatureMapsOffset + rin * inputFeatureMapsShapeW + cin], inputFeatureMaps);//
    }  
  } 
}


