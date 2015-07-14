
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

void atomic_add_global(volatile global float *source, const float operand) {
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
}

#ifdef padding

  __kernel void forwardPass(__global float *inputFeatureMaps,
                            __global float *weights, 
                            __global float *outputFeatureMaps)
{
  int gb0 = get_global_id(0), gb1 = get_global_id(1), gb2 = get_global_id(2);

  bool cond = gb0 < outputFeatureMapH * outputFeatureMapW && gb1 < numOfOutputFeatureMaps && gb2 < batchSize;
  if (cond) {
      int batchOffsetIn = gb2 * numOfInputFeatureMaps * inputFeatureMapH * inputFeatureMapW;
      int batchOffsetOut = gb2 * (numOfOutputFeatureMaps * outputFeatureMapH * outputFeatureMapW);
      int featureMapOffsetOut = batchOffsetOut + gb1 * outputFeatureMapH * outputFeatureMapW;
      int weightsOffsetOut = gb1 * (filterW * filterH * numOfInputFeatureMaps + (addBias ? 1 : 0));
      int rowOut = gb0 / outputFeatureMapW;
      int colOut = gb0 % outputFeatureMapW;
      float out = 0;
      for (int k = 0; k < numOfInputFeatureMaps; k++) {
        int featureMapOffsetIn = batchOffsetIn + k * inputFeatureMapH * inputFeatureMapW;
        int weightsOffsetIn = weightsOffsetOut + k * filterW * filterH;
        for (int r = 0; r < filterH; r++) {
          for (int c = 0; c < filterW; c++) {
            int rowIn = r + rowOut * stride;
            int colIn = c + colOut * stride;
            float preAct = select(0.0f, inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn], rowIn < inputFeatureMapH && colIn < inputFeatureMapW);
            out += preAct * weights[weightsOffsetIn + r * filterW + c];
          }
        }
      }
      if (addBias) {
        out += weights[weightsOffsetOut + filterW * filterH * numOfInputFeatureMaps];
      }
      outputFeatureMaps[featureMapOffsetOut + gb0] = activationFunction(out);
  }
}

 /*   __kernel void backCalcGradients(__global float *inputFeatureMaps,
                                    __global float *errors, 
                                    __global float *gradients)
  {
    int gb0 = get_global_id(0), gb1 = get_global_id(1), gb2 = get_global_id(2);
    int weightsDim = filterW * filterH;
    if (gb0 < weightsDim && gb1 < numOfOutputFeatureMaps && gb2 < numOfInputFeatureMaps) {
      float out = 0;
      float out1 = 0;
      int weightsOffsetOut = gb1 * (weightsDim * numOfInputFeatureMaps + (addBias ? 1 : 0));
      int weightsOffsetIn = weightsOffsetOut + gb2 * weightsDim;
      for (int i = 0; i < batchSize; i++) {
        int batchOffsetIn = i * numOfInputFeatureMaps * inputFeatureMapH * inputFeatureMapW;
        int batchOffsetOut = i * (numOfOutputFeatureMaps * outputFeatureMapH * outputFeatureMapW);
        int featureMapOffsetOut = batchOffsetOut + gb1 * outputFeatureMapH * outputFeatureMapW;
        int featureMapOffsetIn = batchOffsetIn + gb2 * inputFeatureMapH * inputFeatureMapW;
        for (int j = 0; j < outputFeatureMapH * outputFeatureMapW; j++) {
          int rowOut = j / outputFeatureMapW;
          int colOut = j % outputFeatureMapW;
          int rowIn = gb0 / filterW + rowOut * stride;
          int colIn = gb0 % filterW + colOut * stride;
          float act = select(0.0f, inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn], rowIn < inputFeatureMapH && colIn < inputFeatureMapW);
          out += act * errors[featureMapOffsetOut + j];
          if (addBias == 1 && gb2 + 1 == numOfInputFeatureMaps && gb0 == 0) {
            out1 += errors[featureMapOffsetOut + j];
          }
        }
      }
      gradients[weightsOffsetIn + gb0] = out;
      if (addBias == 1 && gb2 + 1 == numOfInputFeatureMaps && gb0 == 0) {
        gradients[weightsOffsetOut + weightsDim * numOfInputFeatureMaps] = out1;
      }
    }
  }*/

  __kernel void backCalcGradients(__global float *inputFeatureMaps,
                                  __global float *errors, 
                                  __global float *gradients)
{
  int lo0 = get_local_id(0), lo1 = get_local_id(1), lo2 = get_local_id(2);
  int gp0 = get_group_id(0), gp1 = get_group_id(1), gp2 = get_group_id(2);
  int gb0 = get_global_id(0), gb1 = get_global_id(1), gb2 = get_global_id(2);
  int weightsDim = filterW * filterH;
  __local float gradients_buff[groupSize_k1_X][groupSize_k1_Y][groupSize_k1_Z][(filterW * filterH)];
  __local float biasGradient[groupSize_k1_X][groupSize_k1_Y];
  for (int j = 0; j < weightsDim; j++) {
    gradients_buff[lo0][lo1][lo2][j] = 0;
  }
  biasGradient[lo0][lo1] = 0;
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  bool cond = gb2 < numOfInputFeatureMaps && gb1 < numOfOutputFeatureMaps && gb0 < batchSize;
  bool cond1 = addBias == 1 && gb2 + 1 == numOfInputFeatureMaps;
  if (cond) {
    int batchOffsetIn = gb0 * numOfInputFeatureMaps * inputFeatureMapH * inputFeatureMapW;
    int batchOffsetOut = gb0 * (numOfOutputFeatureMaps * outputFeatureMapH * outputFeatureMapW);
    int featureMapOffsetIn = batchOffsetIn + gb2 * inputFeatureMapH * inputFeatureMapW;
    int featureMapOffsetOut = batchOffsetOut + gb1 * outputFeatureMapH * outputFeatureMapW;
    int weightsOffsetOut = gb1 * (weightsDim * numOfInputFeatureMaps + (addBias ? 1 : 0));
    int weightsOffsetIn = weightsOffsetOut + gb2 * weightsDim;
    for (int j = 0; j < outputFeatureMapH * outputFeatureMapW; j++) {
      int rowOut = j / outputFeatureMapW;
      int colOut = j % outputFeatureMapW;
      for (int r = 0; r < filterH; r++) {
        for (int c = 0; c < filterW; c++) {
          int weightInd = r * filterW + c;          
          int rowIn = r + rowOut * stride;
          int colIn = c + colOut * stride;
          float act = select(0.0f, inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn], rowIn < inputFeatureMapH && colIn < inputFeatureMapW);
          gradients_buff[lo0][lo1][lo2][weightInd] +=  act * errors[featureMapOffsetOut + j];
        } 
      }
      if (cond1) {
        biasGradient[lo0][lo1] += errors[featureMapOffsetOut + j];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if (lo0 == 0) {
      for (int j = 0; j < weightsDim; j++) {
        float result = 0;
        for (int i = 0; i < groupSize_k1_X && gp0 * groupSize_k1_X + i < batchSize; i++) {
            result += gradients_buff[i][lo1][lo2][j];
        }
        atomic_add_global(&gradients[weightsOffsetIn + j], result);
      }
      if (cond1) { 
        float biasG = 0;
        for (int i = 0; i < groupSize_k1_X && gp0 * groupSize_k1_X + i < batchSize; i++) {
          biasG += biasGradient[i][lo1];
        }
        atomic_add_global(&gradients[weightsOffsetOut + weightsDim * numOfInputFeatureMaps], biasG);
      }

    }
  }
}

/*  __kernel void backCalcGradients(__global float *inputFeatureMaps,
                                  __global float *errors, 
                                  __global float *gradients)
{
  int lo0 = get_local_id(0), lo1 = get_local_id(1), lo2 = get_local_id(2);
  int gp0 = get_group_id(0), gp1 = get_group_id(1), gp2 = get_group_id(2);
  int gb0 = get_global_id(0), gb1 = get_global_id(1), gb2 = get_global_id(2);
  int weightsDim = filterW * filterH;
  __local float gradients_buff[groupSize_k1_Z][groupSize_k1_Y][groupSize_k1_X][numOfInputFeatureMaps];
  __local float biasGradient[groupSize_k1_Z][groupSize_k1_Y];
  for (int j = 0; j < numOfInputFeatureMaps; j++) {
    gradients_buff[lo2][lo1][lo0][j] = 0;
  }
  biasGradient[lo2][lo1] = 0;
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  if (gb0 < weightsDim && gb1 < numOfOutputFeatureMaps && gb2 < batchSize) {
    int batchOffsetIn = gb2 * numOfInputFeatureMaps * inputFeatureMapH * inputFeatureMapW;
    int batchOffsetOut = gb2 * (numOfOutputFeatureMaps * outputFeatureMapH * outputFeatureMapW);
    int featureMapOffsetOut = batchOffsetOut + gb1 * outputFeatureMapH * outputFeatureMapW;
    int weightsOffsetOut = gb1 * (weightsDim * numOfInputFeatureMaps + (addBias ? 1 : 0));
    for (int i = 0; i < numOfInputFeatureMaps; i ++) {
      int featureMapOffsetIn = batchOffsetIn + i * inputFeatureMapH * inputFeatureMapW;
      int weightsOffsetIn = weightsOffsetOut + i * weightsDim;
      for (int j = 0; j < outputFeatureMapH * outputFeatureMapW; j++) {
        int rowOut = j / outputFeatureMapW;
        int colOut = j % outputFeatureMapW;
        int rowIn = gb0 / filterW + rowOut * stride;
        int colIn = gb0 % filterW + colOut * stride;
        if (rowIn < inputFeatureMapH && colIn < inputFeatureMapW) {
          gradients_buff[lo2][lo1][lo0][i] += inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn] * errors[featureMapOffsetOut + j];
        }
        if (addBias == 1 && i + 1 == numOfInputFeatureMaps && gb0 == 0) {
          biasGradient[lo2][lo1] += errors[featureMapOffsetOut + j];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if (lo2 == 0) {
      for (int i = 0; i < numOfInputFeatureMaps; i++) {
        int weightsOffsetIn = weightsOffsetOut + i * weightsDim;
        float result = 0;
        for (int j = 0; j < groupSize_k1_Z && gp2 * groupSize_k1_Z + j < batchSize; j++) {
            result += gradients_buff[j][lo1][lo0][i];
        }
        atomic_add_global(&gradients[weightsOffsetIn + gb0], result);
        if (addBias == 1 && i + 1 == numOfInputFeatureMaps && gb0 == 0) { 
          float biasG = 0;
          for (int i = 0; i < groupSize_k1_Z && gp2 * groupSize_k1_Z + i < batchSize; i++) {
            biasG += biasGradient[i][lo1];
          }
          atomic_add_global(&gradients[weightsOffsetOut + weightsDim * numOfInputFeatureMaps], biasG);
        }
      }
    }
  }
}*/


  //parallel on 3 dimension, seems to be much faster!!!
/*  __kernel void backCalcPrevErr(__global float *weights,
                                    __global float *errors, 
                                    __global float *inputFeatureMaps, 
                                    __global float *prevErrors)
  {
    int lo0 = get_local_id(0), lo1 = get_local_id(1);
    int gb0 = get_global_id(0), gb1 = get_global_id(1), gb2 = get_global_id(2);
    int weightsDim = filterW * filterH;

    bool cond = gb0 < inputFeatureMapH * inputFeatureMapW && gb1 < numOfInputFeatureMaps && gb2 < batchSize;
    if (cond) {
      int batchOffsetIn = gb2 * numOfInputFeatureMaps * inputFeatureMapH * inputFeatureMapW;
      int batchOffsetOut = gb2 * (numOfOutputFeatureMaps * outputFeatureMapH * outputFeatureMapW);
      int featureMapOffsetIn = batchOffsetIn + gb1 * inputFeatureMapH * inputFeatureMapW;
      int rowIn = gb0 / inputFeatureMapW;
      int colIn = gb0 % inputFeatureMapW;
      float der = derivative(inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn]);
      float out = 0;
      int c = (max(0, colIn - filterW + 1) % stride) ? (max(0, colIn - filterW + 1) + stride - max(0, colIn - filterW + 1) % stride) : max(0, colIn - filterW + 1);
      for ( ; c <= colIn; c += stride) {
        int r = (max(0, rowIn - filterH + 1) % stride) ? (max(0, rowIn - filterH + 1) + stride - max(0, rowIn - filterH + 1) % stride) : max(0, rowIn - filterH + 1);
        for ( ; r <= rowIn; r += stride) {
          int rowOut = r / stride;
          int colOut = c / stride;
          int k = (rowIn - r) * filterW + colIn - c;
          for (int j = 0; j < numOfOutputFeatureMaps; j++) {
            int featureMapOffsetOut = batchOffsetOut + j * outputFeatureMapH * outputFeatureMapW;
            int weightsOffsetOut = j * (weightsDim * numOfInputFeatureMaps + (addBias ? 1 : 0));
            int weightsOffsetIn = weightsOffsetOut + gb1 * weightsDim;
            out += weights[weightsOffsetIn + k] * errors[featureMapOffsetOut + rowOut * outputFeatureMapW + colOut] * der;
          }
        }
      }
      prevErrors[featureMapOffsetIn + gb0] = out;
    }
  } */

//using local memory to store weights, faster with big problem
  __kernel void backCalcPrevErr(__global float *weights,
                                    __global float *errors, 
                                    __global float *inputFeatureMaps, 
                                    __global float *prevErrors)
  {
    int lo0 = get_local_id(0), lo1 = get_local_id(1), lo2 = get_local_id(2);
    int gb0 = get_global_id(0), gb1 = get_global_id(1), gb2 = get_global_id(2);
    __local float weights_buff[groupSize_k2_Y][filterW * filterH];
    int weightsDim = filterW * filterH;

    int batchOffsetIn = gb2 * numOfInputFeatureMaps * inputFeatureMapH * inputFeatureMapW;
    int batchOffsetOut = gb2 * (numOfOutputFeatureMaps * outputFeatureMapH * outputFeatureMapW);
    int featureMapOffsetIn = batchOffsetIn + gb1 * inputFeatureMapH * inputFeatureMapW;
    int rowIn = gb0 / inputFeatureMapW;
    int colIn = gb0 % inputFeatureMapW;
    float der = derivative(inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn]);
    float out = 0;
    bool cond = gb0 < inputFeatureMapH * inputFeatureMapW && gb1 < numOfInputFeatureMaps && gb2 < batchSize;
    for (int j = 0; j < numOfOutputFeatureMaps; j++) {
      int weightsOffsetOut = j * (weightsDim * numOfInputFeatureMaps + (addBias ? 1 : 0));
      int weightsOffsetIn = weightsOffsetOut + gb1 * weightsDim;
      for (int c0 = 0; c0 < weightsDim; c0 += groupSize_k2_X * groupSize_k2_Z) {
        if (c0 + lo0 * groupSize_k2_Z + lo2 < weightsDim)
          weights_buff[lo1][c0 + lo0 * groupSize_k2_Z + lo2] = weights[weightsOffsetIn + c0 + lo0 * groupSize_k2_Z + lo2];
      }

      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

      int c = (max(0, colIn - filterW + 1) % stride) ? (max(0, colIn - filterW + 1) + stride - max(0, colIn - filterW + 1) % stride) : max(0, colIn - filterW + 1);
      for ( ; c <= colIn; c += stride) {
        int r = (max(0, rowIn - filterH + 1) % stride) ? (max(0, rowIn - filterH + 1) + stride - max(0, rowIn - filterH + 1) % stride) : max(0, rowIn - filterH + 1);
        for ( ; r <= rowIn; r += stride) {
          int rowOut = r / stride;
          int colOut = c / stride;
          int k = (rowIn - r) * filterW + colIn - c;
          int featureMapOffsetOut = batchOffsetOut + j * outputFeatureMapH * outputFeatureMapW;
          if (cond) {
            out += weights_buff[lo1][k] * errors[featureMapOffsetOut + rowOut * outputFeatureMapW + colOut] * der;
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
    if (cond) {
      prevErrors[featureMapOffsetIn + gb0] = out;
    }
  } 


#else


//kernel not using any local mem, 3D version
__kernel void forwardPass(__global float *inputFeatureMaps,
                            __global float *weights, 
                            __global float *outputFeatureMaps)
{
  int gb0 = get_global_id(0), gb1 = get_global_id(1), gb2 = get_global_id(2);

  bool cond = gb0 < outputFeatureMapH * outputFeatureMapW && gb1 < numOfOutputFeatureMaps && gb2 < batchSize;
  if (cond) {
      int batchOffsetIn = gb2 * numOfInputFeatureMaps * inputFeatureMapH * inputFeatureMapW;
      int batchOffsetOut = gb2 * (numOfOutputFeatureMaps * outputFeatureMapH * outputFeatureMapW);
      int featureMapOffsetOut = batchOffsetOut + gb1 * outputFeatureMapH * outputFeatureMapW;
      int weightsOffsetOut = gb1 * (filterW * filterH * numOfInputFeatureMaps + (addBias ? 1 : 0));
      int rowOut = gb0 / outputFeatureMapW;
      int colOut = gb0 % outputFeatureMapW;
      float out = 0;
      for (int k = 0; k < numOfInputFeatureMaps; k++) {
        int featureMapOffsetIn = batchOffsetIn + k * inputFeatureMapH * inputFeatureMapW;
        int weightsOffsetIn = weightsOffsetOut + k * filterW * filterH;
        for (int r = 0; r < filterH; r++) {
          for (int c = 0; c < filterW; c++) {
            int rowIn = r + rowOut * stride;
            int colIn = c + colOut * stride;
            out += inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn] 
                    * weights[weightsOffsetIn + r * filterW + c];
          }
        }
      }
      if (addBias) {
        out += weights[weightsOffsetOut + filterW * filterH * numOfInputFeatureMaps];
      }
      outputFeatureMaps[featureMapOffsetOut + gb0] = activationFunction(out);
  }
}


 /*   __kernel void backCalcGradients(__global float *inputFeatureMaps,
                                    __global float *errors, 
                                    __global float *gradients)
  {
    int gb0 = get_global_id(0), gb1 = get_global_id(1), gb2 = get_global_id(2);
    int weightsDim = filterW * filterH;
    if (gb0 < weightsDim && gb1 < numOfOutputFeatureMaps && gb2 < numOfInputFeatureMaps) {
      float out = 0;
      float out1 = 0;
      int weightsOffsetOut = gb1 * (weightsDim * numOfInputFeatureMaps + (addBias ? 1 : 0));
      int weightsOffsetIn = weightsOffsetOut + gb2 * weightsDim;
      for (int i = 0; i < batchSize; i++) {
        int batchOffsetIn = i * numOfInputFeatureMaps * inputFeatureMapH * inputFeatureMapW;
        int batchOffsetOut = i * (numOfOutputFeatureMaps * outputFeatureMapH * outputFeatureMapW);
        int featureMapOffsetOut = batchOffsetOut + gb1 * outputFeatureMapH * outputFeatureMapW;
        int featureMapOffsetIn = batchOffsetIn + gb2 * inputFeatureMapH * inputFeatureMapW;
        for (int j = 0; j < outputFeatureMapH * outputFeatureMapW; j++) {
          int rowOut = j / outputFeatureMapW;
          int colOut = j % outputFeatureMapW;
          int rowIn = gb0 / filterW + rowOut * stride;
          int colIn = gb0 % filterW + colOut * stride;
          if (rowIn < inputFeatureMapH && colIn < inputFeatureMapW) {
            out += inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn] 
            * errors[featureMapOffsetOut + j];
          }
          out1 += errors[featureMapOffsetOut + j];
        }
      }
      gradients[weightsOffsetIn + gb0] = out;
      if (addBias == 1 && gb2 + 1 == numOfInputFeatureMaps) {
        gradients[weightsOffsetOut + weightsDim * numOfInputFeatureMaps] = out1;
      }
    }
  }*/

  __kernel void backCalcGradients(__global float *inputFeatureMaps,
                                  __global float *errors, 
                                  __global float *gradients)
{
  int lo0 = get_local_id(0), lo1 = get_local_id(1), lo2 = get_local_id(2);
  int gp0 = get_group_id(0), gp1 = get_group_id(1), gp2 = get_group_id(2);
  int gb0 = get_global_id(0), gb1 = get_global_id(1), gb2 = get_global_id(2);
  int weightsDim = filterW * filterH;
  __local float gradients_buff[groupSize_k1_X][groupSize_k1_Y][groupSize_k1_Z][(filterW * filterH)];
  __local float biasGradient[groupSize_k1_X][groupSize_k1_Y];
  for (int j = 0; j < weightsDim; j++) {
    gradients_buff[lo0][lo1][lo2][j] = 0;
  }
  biasGradient[lo0][lo1] = 0;
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  bool cond = gb2 < numOfInputFeatureMaps && gb1 < numOfOutputFeatureMaps && gb0 < batchSize;
  bool cond1 = addBias == 1 && gb2 + 1 == numOfInputFeatureMaps;
  if (cond) {
    int batchOffsetIn = gb0 * numOfInputFeatureMaps * inputFeatureMapH * inputFeatureMapW;
    int batchOffsetOut = gb0 * (numOfOutputFeatureMaps * outputFeatureMapH * outputFeatureMapW);
    int featureMapOffsetIn = batchOffsetIn + gb2 * inputFeatureMapH * inputFeatureMapW;
    int featureMapOffsetOut = batchOffsetOut + gb1 * outputFeatureMapH * outputFeatureMapW;
    int weightsOffsetOut = gb1 * (weightsDim * numOfInputFeatureMaps + (addBias ? 1 : 0));
    int weightsOffsetIn = weightsOffsetOut + gb2 * weightsDim;
    for (int j = 0; j < outputFeatureMapH * outputFeatureMapW; j++) {
      int rowOut = j / outputFeatureMapW;
      int colOut = j % outputFeatureMapW;
      for (int r = 0; r < filterH; r++) {
        for (int c = 0; c < filterW; c++) {
          int weightInd = r * filterW + c;          
          int rowIn = r + rowOut * stride;
          int colIn = c + colOut * stride;
          gradients_buff[lo0][lo1][lo2][weightInd] += inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn] * errors[featureMapOffsetOut + j];
        } 
      }
      if (cond1) {
        biasGradient[lo0][lo1] += errors[featureMapOffsetOut + j];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if (lo0 == 0) {
      for (int j = 0; j < weightsDim; j++) {
        float result = 0;
        for (int i = 0; i < groupSize_k1_X && gp0 * groupSize_k1_X + i < batchSize; i++) {
            result += gradients_buff[i][lo1][lo2][j];
        }
        atomic_add_global(&gradients[weightsOffsetIn + j], result);
      }
      if (cond1) { 
        float biasG = 0;
        for (int i = 0; i < groupSize_k1_X && gp0 * groupSize_k1_X + i < batchSize; i++) {
          biasG += biasGradient[i][lo1];
        }
        atomic_add_global(&gradients[weightsOffsetOut + weightsDim * numOfInputFeatureMaps], biasG);
      }

    }
  }
}

  /*__kernel void backCalcGradients(__global float *inputFeatureMaps,
                                  __global float *errors, 
                                  __global float *gradients)
{
  int lo0 = get_local_id(0), lo1 = get_local_id(1), lo2 = get_local_id(2);
  int gp0 = get_group_id(0), gp1 = get_group_id(1), gp2 = get_group_id(2);
  int gb0 = get_global_id(0), gb1 = get_global_id(1), gb2 = get_global_id(2);
  int weightsDim = filterW * filterH;
  __local float gradients_buff[groupSize_k1_Z][groupSize_k1_Y][groupSize_k1_X][numOfInputFeatureMaps];
  __local float biasGradient[groupSize_k1_Z][groupSize_k1_Y];
  for (int j = 0; j < numOfInputFeatureMaps; j++) {
    gradients_buff[lo2][lo1][lo0][j] = 0;
  }
  biasGradient[lo2][lo1] = 0;
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  if (gb0 < weightsDim && gb1 < numOfOutputFeatureMaps && gb2 < batchSize) {
    int batchOffsetIn = gb2 * numOfInputFeatureMaps * inputFeatureMapH * inputFeatureMapW;
    int batchOffsetOut = gb2 * (numOfOutputFeatureMaps * outputFeatureMapH * outputFeatureMapW);
    int featureMapOffsetOut = batchOffsetOut + gb1 * outputFeatureMapH * outputFeatureMapW;
    int weightsOffsetOut = gb1 * (weightsDim * numOfInputFeatureMaps + (addBias ? 1 : 0));
    for (int i = 0; i < numOfInputFeatureMaps; i ++) {
      int featureMapOffsetIn = batchOffsetIn + i * inputFeatureMapH * inputFeatureMapW;
      int weightsOffsetIn = weightsOffsetOut + i * weightsDim;
      for (int j = 0; j < outputFeatureMapH * outputFeatureMapW; j++) {
        int rowOut = j / outputFeatureMapW;
        int colOut = j % outputFeatureMapW;
        int rowIn = gb0 / filterW + rowOut * stride;
        int colIn = gb0 % filterW + colOut * stride;
        gradients_buff[lo2][lo1][lo0][i] += inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn] * errors[featureMapOffsetOut + j];
        if (addBias == 1 && i + 1 == numOfInputFeatureMaps && gb0 == 0) {
          biasGradient[lo2][lo1] += errors[featureMapOffsetOut + j];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if (lo2 == 0) {
      for (int i = 0; i < numOfInputFeatureMaps; i++) {
        int weightsOffsetIn = weightsOffsetOut + i * weightsDim;
        float result = 0;
        for (int j = 0; j < groupSize_k1_Z && gp2 * groupSize_k1_Z + j < batchSize; j++) {
            result += gradients_buff[j][lo1][lo0][i];
        }
        atomic_add_global(&gradients[weightsOffsetIn + gb0], result);
        if (addBias == 1 && i + 1 == numOfInputFeatureMaps && gb0 == 0) { 
          float biasG = 0;
          for (int i = 0; i < groupSize_k1_Z && gp2 * groupSize_k1_Z + i < batchSize; i++) {
            biasG += biasGradient[i][lo1];
          }
          atomic_add_global(&gradients[weightsOffsetOut + weightsDim * numOfInputFeatureMaps], biasG);
        }
      }
    }
  }
}*/



//parallel on 3 dimension, seems to be much faster!!!
/*__kernel void backCalcPrevErr(__global float *weights,
                                  __global float *errors, 
                                  __global float *inputFeatureMaps, 
                                  __global float *prevErrors)
{
  int lo0 = get_local_id(0), lo1 = get_local_id(1);
  int gb0 = get_global_id(0), gb1 = get_global_id(1), gb2 = get_global_id(2);
  int weightsDim = filterW * filterH;

  bool cond = gb0 < inputFeatureMapH * inputFeatureMapW && gb1 < numOfInputFeatureMaps && gb2 < batchSize;
  if (cond) {
    int batchOffsetIn = gb2 * numOfInputFeatureMaps * inputFeatureMapH * inputFeatureMapW;
    int batchOffsetOut = gb2 * (numOfOutputFeatureMaps * outputFeatureMapH * outputFeatureMapW);
    int featureMapOffsetIn = batchOffsetIn + gb1 * inputFeatureMapH * inputFeatureMapW;
    int rowIn = gb0 / inputFeatureMapW;
    int colIn = gb0 % inputFeatureMapW;
    float der = derivative(inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn]);
    float out = 0;
    int c = (max(0, colIn - filterW + 1) % stride) ? (max(0, colIn - filterW + 1) + stride - max(0, colIn - filterW + 1) % stride) : max(0, colIn - filterW + 1);
    for ( ; c <= colIn && c + filterW <= inputFeatureMapW; c += stride) {
      int r = (max(0, rowIn - filterH + 1) % stride) ? (max(0, rowIn - filterH + 1) + stride - max(0, rowIn - filterH + 1) % stride) : max(0, rowIn - filterH + 1);
      for ( ; r <= rowIn && r + filterH <= inputFeatureMapH; r += stride) {
        int rowOut = r / stride;
        int colOut = c / stride;
        int k = (rowIn - r) * filterW + colIn - c;
        for (int j = 0; j < numOfOutputFeatureMaps; j++) {
          int featureMapOffsetOut = batchOffsetOut + j * outputFeatureMapH * outputFeatureMapW;
          int weightsOffsetOut = j * (weightsDim * numOfInputFeatureMaps + (addBias ? 1 : 0));
          int weightsOffsetIn = weightsOffsetOut + gb1 * weightsDim;
          out += weights[weightsOffsetIn + k] * errors[featureMapOffsetOut + rowOut * outputFeatureMapW + colOut] * der;
        }
      }
    }
    prevErrors[featureMapOffsetIn + gb0] = out;
  }
} */

//using local memory to store weights, faster with big problem
  __kernel void backCalcPrevErr(__global float *weights,
                                    __global float *errors, 
                                    __global float *inputFeatureMaps, 
                                    __global float *prevErrors)
  {
    int lo0 = get_local_id(0), lo1 = get_local_id(1), lo2 = get_local_id(2);
    int gb0 = get_global_id(0), gb1 = get_global_id(1), gb2 = get_global_id(2);
    __local float weights_buff[groupSize_k2_Y][filterW * filterH];
    int weightsDim = filterW * filterH;

    int batchOffsetIn = gb2 * numOfInputFeatureMaps * inputFeatureMapH * inputFeatureMapW;
    int batchOffsetOut = gb2 * (numOfOutputFeatureMaps * outputFeatureMapH * outputFeatureMapW);
    int featureMapOffsetIn = batchOffsetIn + gb1 * inputFeatureMapH * inputFeatureMapW;
    int rowIn = gb0 / inputFeatureMapW;
    int colIn = gb0 % inputFeatureMapW;
    float der = derivative(inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn]);
    float out = 0;
    bool cond = gb0 < inputFeatureMapH * inputFeatureMapW && gb1 < numOfInputFeatureMaps && gb2 < batchSize;
    for (int j = 0; j < numOfOutputFeatureMaps; j++) {
      int weightsOffsetOut = j * (weightsDim * numOfInputFeatureMaps + (addBias ? 1 : 0));
      int weightsOffsetIn = weightsOffsetOut + gb1 * weightsDim;
      for (int c0 = 0; c0 < weightsDim; c0 += groupSize_k2_X * groupSize_k2_Z) {
        if (c0 + lo0 * groupSize_k2_Z + lo2 < weightsDim)
          weights_buff[lo1][c0 + lo0 * groupSize_k2_Z + lo2] = weights[weightsOffsetIn + c0 + lo0 * groupSize_k2_Z + lo2];
      }

      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

      int c = (max(0, colIn - filterW + 1) % stride) ? (max(0, colIn - filterW + 1) + stride - max(0, colIn - filterW + 1) % stride) : max(0, colIn - filterW + 1);
      for ( ; c <= colIn && c + filterW <= inputFeatureMapW; c += stride) {
        int r = (max(0, rowIn - filterH + 1) % stride) ? (max(0, rowIn - filterH + 1) + stride - max(0, rowIn - filterH + 1) % stride) : max(0, rowIn - filterH + 1);
        for ( ; r <= rowIn && r + filterH <= inputFeatureMapH; r += stride) {
          int rowOut = r / stride;
          int colOut = c / stride;
          int k = (rowIn - r) * filterW + colIn - c;
          int featureMapOffsetOut = batchOffsetOut + j * outputFeatureMapH * outputFeatureMapW;
          if (cond) {
            out += weights_buff[lo1][k] * errors[featureMapOffsetOut + rowOut * outputFeatureMapW + colOut] * der;
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
    if (cond) {
      prevErrors[featureMapOffsetIn + gb0] = out;
    }
  } 


#endif

__kernel void updateWeights(__global float *gradients, __global float *weights, __global float *weightsUpdate, float learningRate, float momentum, float weightDecay_batchSize, int len)
{
    int b0 = get_group_id(0);
    int t0 = get_local_id(0);
    float private_weights[1];
    float private_weightsUpdate[1];

    float lr = learningRate;
    float decay = weightDecay_batchSize;

    for (int c0 = 128 * b0; c0 < len; c0 += 32768) {
     if ((c0 + t0) % (numOfInputFeatureMaps * filterW * filterH + (addBias ? 1 : 0)) == numOfInputFeatureMaps * filterW * filterH ) {
        lr = 2 * learningRate;
        decay = 0;
     }
     if (len >= t0 + c0 + 1) {
        private_weights[0] = weights[t0 + c0];
        private_weightsUpdate[0] = weightsUpdate[t0 + c0];
        private_weightsUpdate[0] = (((momentum * private_weightsUpdate[0]) - (lr * gradients[t0 + c0])) - ((lr * decay) * private_weights[0]));
        private_weights[0] += private_weightsUpdate[0];
        weightsUpdate[t0 + c0] = private_weightsUpdate[0];
        weights[t0 + c0] = private_weights[0];
      }
    }
}


 