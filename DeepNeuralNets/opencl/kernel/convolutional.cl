
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


__kernel void forwardPass(__global float *inputFeatureMaps,
                            __global float *weights, 
                            __global float *outputFeatureMaps)
{
  int lo0 = get_local_id(0), lo1 = get_local_id(1), lo2 = get_local_id(2);
  int gb0 = get_global_id(0), gb1 = get_global_id(1), gb2 = get_global_id(2);
  __local float weights_buff[groupSize_k0_Y][filterW * filterH];

  bool cond = gb0 < outputFeatureMapH * outputFeatureMapW && gb1 < numOfOutputFeatureMaps && gb2 < batchSize;
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
    for (int c0 = 0; c0 < filterW * filterH; c0 += groupSize_k0_X * groupSize_k0_Z) {
      if (c0 + lo0 * groupSize_k0_Z + lo2 < filterW * filterH)
        weights_buff[lo1][c0 + lo0 * groupSize_k0_Z + lo2] = weights[weightsOffsetIn + c0 + lo0 * groupSize_k0_Z + lo2];
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    for (int r = 0; r < filterH; r++) {
      for (int c = 0; c < filterW; c++) {
        #ifdef padding
        int rowIn = r + rowOut * stride - (filterH - 1) / 2;
        int colIn = c + colOut * stride - (filterW - 1) / 2;
        int cond1 = rowIn < inputFeatureMapH && colIn < inputFeatureMapW && rowIn >=0 && colIn >=0;
        float preAct = select(0.0f, inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn], cond1);
        #else
        int rowIn = r + rowOut * stride;
        int colIn = c + colOut * stride;
        float preAct = inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn];
        #endif
        if (cond) {
          out += preAct * weights_buff[lo1][r * filterW + c];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }
  if (cond) {
    out = select(out, out + weights[weightsOffsetOut + filterW * filterH * numOfInputFeatureMaps], addBias);
    outputFeatureMaps[featureMapOffsetOut + gb0] = activationFunction(out);
  }
}


 

#define groupSize_k1_chunk 56
  __kernel void backCalcGradients(__global float *inputFeatureMaps,
                                  __global float *errors, 
                                  __global float *gradients)
{
  int lo0 = get_local_id(0), lo1 = get_local_id(1), lo2 = get_local_id(2);
  int gp0 = get_group_id(0), gp1 = get_group_id(1), gp2 = get_group_id(2);
  int gb0 = get_global_id(0), gb1 = get_global_id(1), gb2 = get_global_id(2);
  int weightsDim = filterW * filterH;
  __local float input_buff[groupSize_k1_Z][groupSize_k1_X][groupSize_k1_chunk];
  __local float error_buff[groupSize_k1_Y][groupSize_k1_chunk];
  bool cond = gb0 < weightsDim && gb1 < numOfOutputFeatureMaps && gb2 < numOfInputFeatureMaps;
  float out = 0;
  float out1 = 0;
  int weightsOffsetOut = gb1 * (weightsDim * numOfInputFeatureMaps + (addBias ? 1 : 0));
  int weightsOffsetIn = weightsOffsetOut + gb2 * weightsDim;
  for (int i = 0; i < batchSize; i++) {
    int batchOffsetIn = i * numOfInputFeatureMaps * inputFeatureMapH * inputFeatureMapW;
    int batchOffsetOut = i * (numOfOutputFeatureMaps * outputFeatureMapH * outputFeatureMapW);
    int featureMapOffsetOut = batchOffsetOut + gb1 * outputFeatureMapH * outputFeatureMapW;
    int featureMapOffsetIn = batchOffsetIn + gb2 * inputFeatureMapH * inputFeatureMapW;
    for (int k = 0; k < outputFeatureMapH * outputFeatureMapW; k += groupSize_k1_chunk) {
      for (int c = 0; c + lo2 * groupSize_k1_X + lo0 < groupSize_k1_chunk; c += groupSize_k1_X * groupSize_k1_Z) {
        int j = k + c + lo2 * groupSize_k1_X + lo0;
        error_buff[lo1][c + lo2 * groupSize_k1_X + lo0] = errors[featureMapOffsetOut + j];
      }
      for (int c = 0; c + lo1 < groupSize_k1_chunk; c += groupSize_k1_Y) {
        // int j = k + c + lo1;
        int rowOut = (k + c + lo1) / outputFeatureMapW;
        int colOut = (k + c + lo1) % outputFeatureMapW;
        #ifdef padding
        int rowIn = gb0 / filterW + rowOut * stride - (filterH - 1) / 2;
        int colIn = gb0 % filterW + colOut * stride - (filterW - 1) / 2;
        int cond2 = rowIn < inputFeatureMapH && colIn < inputFeatureMapW && rowIn >=0 && colIn >=0;
        input_buff[lo2][lo0][c + lo1] = select(0.0f, inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn], cond2);
        #else
        int rowIn = gb0 / filterW + rowOut * stride;
        int colIn = gb0 % filterW + colOut * stride;
        input_buff[lo2][lo0][c + lo1] = inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn];
        #endif
      }

      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      // #pragma unroll 16
      for (int j = 0; j < groupSize_k1_chunk; j++) {
        int cond1 = j + k < outputFeatureMapH * outputFeatureMapW && cond;
        out = select(out, out + input_buff[lo2][lo0][j] * error_buff[lo1][j], cond1);
        out1 = select(out1, out1 + error_buff[lo1][j], cond1);
      }
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
  }
  if (cond) {
    gradients[weightsOffsetIn + gb0] = out;
    if (addBias == 1 && gb2 + 1 == numOfInputFeatureMaps) {
      gradients[weightsOffsetOut + weightsDim * numOfInputFeatureMaps] = out1;
    }
  }
}





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

      #ifdef padding
      //TODO
      int r1 = max(-(filterH - 1) / 2, rowIn - filterH + 1);
      int r = ((r1 + (filterH - 1) / 2) % stride) ? (r1 + stride - (r1 + (filterH - 1) / 2) % stride) : r1;
      for ( ; r <= rowIn && r + filterH <= inputFeatureMapH + (filterH - 1) - (filterH - 1) / 2; r += stride) {
        int c1 = max(-(filterW - 1) / 2, colIn - filterW + 1);
        int c = ((c1 + (filterW - 1) / 2) % stride) ? (c1 + stride - (c1 + (filterW - 1) / 2) % stride) : c1;
        for ( ; c <= colIn && c + filterW <= inputFeatureMapW + (filterW - 1) - (filterW - 1) / 2; c += stride) {
          int rowOut = (r + (filterH - 1) / 2) / stride;
          int colOut = (c + (filterW - 1) / 2) / stride;
          int k = (rowIn - r) * filterW + colIn - c;
          int featureMapOffsetOut = batchOffsetOut + j * outputFeatureMapH * outputFeatureMapW;
          if (cond) {
            out += weights_buff[lo1][k] * errors[featureMapOffsetOut + rowOut * outputFeatureMapW + colOut] * der;
          }
        }
      }
      #else
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
      #endif
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
    if (cond) {
      prevErrors[featureMapOffsetIn + gb0] = out;
    }
  } 


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


