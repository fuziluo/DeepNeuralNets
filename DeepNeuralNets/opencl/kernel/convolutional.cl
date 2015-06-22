
#define min(x,y)    ((x) < (y) ? (x) : (y))
#define max(x,y)    ((x) > (y) ? (x) : (y))
float activationFunction(float input) {
  float output = (float) (1/(1 + exp(-input)));
  return output;
}
float derivative(float input) {
    return input * (1 - input);//TODO sigmoid only
}
//kernel not using any local mem
/*__kernel void forwardPass(__global float *inputFeatureMaps,
                            __global float *weights, 
                            __global float *outputFeatureMaps)
{
  int gb0 = get_global_id(0), gb1 = get_global_id(1);
  int weightsDim = filterW * filterH + (addBias ? 1 : 0);

  // if (gb0 < outputFeatureMapH * outputFeatureMapW && gb1 < numOfOutputFeatureMaps) {
    for (int i = 0; i < batchSize; i++) {
      int batchOffsetIn = i * numOfInputFeatureMaps * inputFeatureMapH * inputFeatureMapW;
      int batchOffsetOut = i * (numOfOutputFeatureMaps * outputFeatureMapH * outputFeatureMapW);
      int featureMapOffsetOut = batchOffsetOut + gb1 * outputFeatureMapH * outputFeatureMapW;
      int rowOut = gb0 / outputFeatureMapW;
      int colOut = gb0 % outputFeatureMapW;
      for (int k = 0; k < numOfInputFeatureMaps; k++) {
        int featureMapOffsetIn = batchOffsetIn + k * inputFeatureMapH * inputFeatureMapW;
        for (int m = 0; m < filterW * filterH; m++) {
          int rowIn = m / filterW + rowOut * stride;
          int colIn = m % filterW + colOut * stride;
          outputFeatureMaps[featureMapOffsetOut + rowOut * outputFeatureMapW + colOut] += 
              inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn] * weights[gb1 *  weightsDim + m];
        }
        if (addBias) {
          outputFeatureMaps[featureMapOffsetOut + rowOut * outputFeatureMapW + colOut] += weights[gb1 *  weightsDim + weightsDim - 1];
        }
      }
      outputFeatureMaps[featureMapOffsetOut + rowOut * outputFeatureMapW + colOut] = activationFunction(outputFeatureMaps[featureMapOffsetOut + rowOut * outputFeatureMapW + colOut]);
    }
  // }
}*/

//kernel using local mem to store weights and groupSize_k0_K to partition buffer of weights
/*  __kernel void forwardPass(__global float *inputFeatureMaps,
                            __global float *weights, 
                            __global float *outputFeatureMaps)
{
  int gp0 = get_group_id(0), gp1 = get_group_id(1);
  int lo0 = get_local_id(0), lo1 = get_local_id(1);
  int gb0 = get_global_id(0), gb1 = get_global_id(1);
  __local float shared_W[groupSize_k0_N * groupSize_k0_K];

  int weightsDim = filterW * filterH + (addBias ? 1 : 0);

  if (gb1 < numOfOutputFeatureMaps ) {
    for (int i = 0; i < batchSize; i++) {
      int batchOffsetIn = i * numOfInputFeatureMaps * inputFeatureMapH * inputFeatureMapW;
      int batchOffsetOut = i * (numOfOutputFeatureMaps * outputFeatureMapH * outputFeatureMapW);
      int featureMapOffsetOut = batchOffsetOut + gb1 * outputFeatureMapH * outputFeatureMapW;
      int rowOut = gb0 / outputFeatureMapW;
      int colOut = gb0 % outputFeatureMapW;
      float out = 0;
      for(int j = 0; j < weightsDim; j += groupSize_k0_K) {
        for(int c0 = lo0; j + c0 < weightsDim && c0 < groupSize_k0_K; c0 += groupSize_k0_M) {
          shared_W[lo1 * groupSize_k0_K + c0] = weights[gb1 * weightsDim + j + c0];
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (gb0 < outputFeatureMapH * outputFeatureMapW) {
          for (int k = 0; k < numOfInputFeatureMaps; k++) {
            int featureMapOffsetIn = batchOffsetIn + k * inputFeatureMapH * inputFeatureMapW;
            for (int l = 0; l < groupSize_k0_K && l < filterW * filterH - j; l++) {
              int m = l + j;
              int rowIn = m / filterW + rowOut * stride;
              int colIn = m % filterW + colOut * stride;
              out += 
                  inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn] * shared_W[lo1 * groupSize_k0_K + l];
            }
            if (addBias == 1 && (filterW * filterH - j <= groupSize_k0_K)) {
              out += shared_W[lo1 * groupSize_k0_K + weightsDim - j - 1];
            }
          }
        }
      }
      out = activationFunction(out);
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      if (gb0 < outputFeatureMapH * outputFeatureMapW)
        outputFeatureMaps[featureMapOffsetOut + rowOut * outputFeatureMapW + colOut] = out;
    }
  }
}*/

//kernel using local mem to store weights
  __kernel void forwardPass(__global float *inputFeatureMaps,
                            __global float *weights, 
                            __global float *outputFeatureMaps)
{
  int lo0 = get_local_id(0), lo1 = get_local_id(1);
  int gb0 = get_global_id(0), gb1 = get_global_id(1);
  __local float shared_W[groupSize_k0_N * (filterW * filterH + (addBias ? 1 : 0))];
  int weightsDim = filterW * filterH + (addBias ? 1 : 0);

  if (gb1 < numOfOutputFeatureMaps) {
    for (int i = 0; i < batchSize; i++) {
      int batchOffsetIn = i * numOfInputFeatureMaps * inputFeatureMapH * inputFeatureMapW;
      int batchOffsetOut = i * (numOfOutputFeatureMaps * outputFeatureMapH * outputFeatureMapW);
      int featureMapOffsetOut = batchOffsetOut + gb1 * outputFeatureMapH * outputFeatureMapW;
      int rowOut = gb0 / outputFeatureMapW;
      int colOut = gb0 % outputFeatureMapW;
      float out = 0;
        for(int c0 = lo0; c0 < weightsDim; c0 += groupSize_k0_M) {
          shared_W[lo1 * weightsDim + c0] = weights[gb1 * weightsDim + c0];
        }
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      if (gb0 < outputFeatureMapH * outputFeatureMapW) {
        for (int k = 0; k < numOfInputFeatureMaps; k++) {
          int featureMapOffsetIn = batchOffsetIn + k * inputFeatureMapH * inputFeatureMapW;
          for (int l = 0; l < weightsDim && l < filterW * filterH; l++) {
            int rowIn = l / filterW + rowOut * stride;
            int colIn = l % filterW + colOut * stride;
            out += inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn] * shared_W[lo1 * weightsDim + l];
          }
          if (addBias == 1 && (filterW * filterH <= weightsDim)) {
            out += shared_W[lo1 * weightsDim + weightsDim - 1];
          }
        }
      }
      out = activationFunction(out);
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      if (gb0 < outputFeatureMapH * outputFeatureMapW) {
        outputFeatureMaps[featureMapOffsetOut + rowOut * outputFeatureMapW + colOut] = out;
      }
    }
  }
}

__kernel void backCalcGradients(__global float *inputFeatureMaps,
                                  __global float *errors, 
                                  __global float *gradients)
{
  // int lo0 = get_local_id(0), lo1 = get_local_id(1);
  int gb0 = get_global_id(0), gb1 = get_global_id(1);
  // __local float shared_E[groupSize_k0_N * (filterW * filterH + (addBias ? 1 : 0))];
  int weightsDim = filterW * filterH + (addBias ? 1 : 0);
  if (gb0 < weightsDim && gb1 < numOfOutputFeatureMaps) {
    float out = 0;
    for (int i = 0; i < batchSize; i++) {
      int batchOffsetIn = i * numOfInputFeatureMaps * inputFeatureMapH * inputFeatureMapW;
      int batchOffsetOut = i * (numOfOutputFeatureMaps * outputFeatureMapH * outputFeatureMapW);
      int featureMapOffsetOut = batchOffsetOut + gb1 * outputFeatureMapH * outputFeatureMapW;
      for (int j = 0; j < outputFeatureMapH * outputFeatureMapW; j++) {
        int rowOut = j / outputFeatureMapW;
        int colOut = j % outputFeatureMapW;
        for (int k = 0; k < numOfInputFeatureMaps; k++) {
          int featureMapOffsetIn = batchOffsetIn + k * inputFeatureMapH * inputFeatureMapW;
          if (gb0 < weightsDim - 1) {
            int rowIn = gb0 / filterW + rowOut * stride;
            int colIn = gb0 % filterW + colOut * stride;
            out += inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn] 
                                                * errors[featureMapOffsetOut + rowOut * outputFeatureMapW + colOut] / batchSize;
          } else if (addBias == 1) {
            out += errors[featureMapOffsetOut + rowOut * outputFeatureMapW + colOut] / batchSize;
          }
        }
      }
    }
    // barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    gradients[gb1 * weightsDim + gb0] = out;
  }
}

// incomplete, need to use atomic operation
// __kernel void backCalcGradients(__global float *inputFeatureMaps,
//                                   __global float *errors, 
//                                   __global float *gradients)
// {
//   // int lo0 = get_local_id(0), lo1 = get_local_id(1);
//   int gb0 = get_global_id(0), gb1 = get_global_id(1);
//   // __local float shared_E[groupSize_k0_N * (filterW * filterH + (addBias ? 1 : 0))];
//   int weightsDim = filterW * filterH + (addBias ? 1 : 0);
//   if (gb0 < outputFeatureMapH * outputFeatureMapW && gb1 < numOfOutputFeatureMaps) {
//     for (int i = 0; i < batchSize; i++) {
//       int batchOffsetIn = i * numOfInputFeatureMaps * inputFeatureMapH * inputFeatureMapW;
//       int batchOffsetOut = i * (numOfOutputFeatureMaps * outputFeatureMapH * outputFeatureMapW);
//       int featureMapOffsetOut = batchOffsetOut + gb1 * outputFeatureMapH * outputFeatureMapW;
//       int rowOut = gb0 / outputFeatureMapW;
//       int colOut = gb0 % outputFeatureMapW;
//       for (int k = 0; k < numOfInputFeatureMaps; k++) {
//         int featureMapOffsetIn = batchOffsetIn + k * inputFeatureMapH * inputFeatureMapW;
//         for (int l = 0; l < filterW * filterH; l++) {
//           int rowIn = l / filterW + rowOut * stride;
//           int colIn = l % filterW + colOut * stride;
//           gradients[gb1 * weightsDim + l] += inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn] 
//                                               * errors[featureMapOffsetOut + rowOut * outputFeatureMapW + colOut] / batchSize;
//         }
//         if (addBias == 1) {
//           gradients[gb1 * weightsDim + weightsDim - 1] += errors[featureMapOffsetOut + rowOut * outputFeatureMapW + colOut] / batchSize;
//         }
//       }
//     }
//   }
// }

// first dimension of work size on filterW * filterH, need to initialize prevErrors to zero
/*__kernel void backCalcPrevErr(__global float *weights,
                                  __global float *errors, 
                                  __global float *inputFeatureMaps, 
                                  __global float *prevErrors)
{
  int lo0 = get_local_id(0), lo1 = get_local_id(1);
  int gb0 = get_global_id(0), gb1 = get_global_id(1);
  int weightsDim = filterW * filterH + (addBias ? 1 : 0);
  if (gb0 < filterW * filterH && gb1 < numOfInputFeatureMaps) {
    for (int i = 0; i < batchSize; i++) {
      int batchOffsetIn = i * numOfInputFeatureMaps * inputFeatureMapH * inputFeatureMapW;
      int batchOffsetOut = i * (numOfOutputFeatureMaps * outputFeatureMapH * outputFeatureMapW);
      int featureMapOffsetIn = batchOffsetIn + gb1 * inputFeatureMapH * inputFeatureMapW;
      //TODO clear input first
      for (int k = 0; k < outputFeatureMapH * outputFeatureMapW; k++) {
        int rowOut = k / outputFeatureMapW;
        int colOut = k % outputFeatureMapW;
        int rowIn = gb0 / filterW + rowOut * stride;
        int colIn = gb0 % filterW + colOut * stride;
        float der = derivative(inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn]);
        float out = prevErrors[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn];
        for (int j = 0; j < numOfOutputFeatureMaps; j++) {
          int featureMapOffsetOut = batchOffsetOut + j * outputFeatureMapH * outputFeatureMapW;
          out +=  weights[j * weightsDim + gb0] * errors[featureMapOffsetOut + rowOut * outputFeatureMapW + colOut] * der;
        }
        // barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);  
        prevErrors[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn] = out;
      }
    }
  }
} */



// first dimension of work size on outputFeatureMapH * outputFeatureMapW, need to initialize prevErrors to zero
/*__kernel void backCalcPrevErr(__global float *weights,
                                  __global float *errors, 
                                  __global float *inputFeatureMaps, 
                                  __global float *prevErrors)
{
  int lo0 = get_local_id(0), lo1 = get_local_id(1);
  int gb0 = get_global_id(0), gb1 = get_global_id(1);
  int weightsDim = filterW * filterH + (addBias ? 1 : 0);
  if (gb0 < outputFeatureMapH * outputFeatureMapW && gb1 < numOfInputFeatureMaps) {
    for (int i = 0; i < batchSize; i++) {
      int batchOffsetIn = i * numOfInputFeatureMaps * inputFeatureMapH * inputFeatureMapW;
      int batchOffsetOut = i * (numOfOutputFeatureMaps * outputFeatureMapH * outputFeatureMapW);
      int featureMapOffsetIn = batchOffsetIn + gb1 * inputFeatureMapH * inputFeatureMapW;
      //TODO clear input first
      for (int k = 0; k < filterH * filterW; k++) {
        int rowOut = gb0 / outputFeatureMapW;
        int colOut = gb0 % outputFeatureMapW;
        int rowIn = k / filterW + rowOut * stride;
        int colIn = k % filterW + colOut * stride;
        float der = derivative(inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn]);
        float out = prevErrors[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn];
        for (int j = 0; j < numOfOutputFeatureMaps; j++) {
          int featureMapOffsetOut = batchOffsetOut + j * outputFeatureMapH * outputFeatureMapW;
          out +=  weights[j * weightsDim + k] * errors[featureMapOffsetOut + rowOut * outputFeatureMapW + colOut] * der;
        }
        // barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);  
        prevErrors[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn] = out;
      }
    }
  }
} */

// first dimension of work size on inputFeatureMapH * inputFeatureMapW, does not need to initialize prevErrors to zero
__kernel void backCalcPrevErr(__global float *weights,
                                  __global float *errors, 
                                  __global float *inputFeatureMaps, 
                                  __global float *prevErrors)
{
  int lo0 = get_local_id(0), lo1 = get_local_id(1);
  int gb0 = get_global_id(0), gb1 = get_global_id(1);
  int weightsDim = filterW * filterH + (addBias ? 1 : 0);
  if (gb0 < inputFeatureMapH * inputFeatureMapW && gb1 < numOfInputFeatureMaps) {
    for (int i = 0; i < batchSize; i++) {
      int batchOffsetIn = i * numOfInputFeatureMaps * inputFeatureMapH * inputFeatureMapW;
      int batchOffsetOut = i * (numOfOutputFeatureMaps * outputFeatureMapH * outputFeatureMapW);
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
            out += weights[j * weightsDim + k] * errors[featureMapOffsetOut + rowOut * outputFeatureMapW + colOut] * der;
          }
        }
      }
      // barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);  
      prevErrors[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn] = out;
    }
  }
} 

__kernel void updateWeights(__global float *gradients, __global float *weights, __global float *weightsUpdate, float learningRate, float momentum, float weightDecay_batchSize, int len)
{
    int b0 = get_group_id(0);
    int t0 = get_local_id(0);
    float private_weights[1];
    float private_weightsUpdate[1];

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    for (int c0 = 128 * b0; c0 < len; c0 += 32768)
      if (len >= t0 + c0 + 1) {
        private_weights[0] = weights[t0 + c0];
        private_weightsUpdate[0] = weightsUpdate[t0 + c0];
        private_weightsUpdate[0] = (((momentum * private_weightsUpdate[0]) - (learningRate * gradients[t0 + c0])) - ((learningRate * weightDecay_batchSize) * private_weights[0]));
        private_weights[0] += private_weightsUpdate[0];
        weightsUpdate[t0 + c0] = private_weightsUpdate[0];
        weights[t0 + c0] = private_weights[0];
      }
}


 //kernel using local mem to store weights and groupSize_k0_K to partition buffer of input feature maps
/*  __kernel void forwardPass(__global float *inputFeatureMaps,
                            __global float *weights, 
                            __global float *outputFeatureMaps)
{
  int gp0 = get_group_id(0), gp1 = get_group_id(1);
  int lo0 = get_local_id(0), lo1 = get_local_id(1);
  int gb0 = get_global_id(0), gb1 = get_global_id(1);
  //shared buffer storing part of input feature map data
  __local float shared_I[groupSize_k0_K][inputFeatureMapW]; 
  __local float shared_W[groupSize_k0_N * (filterW * filterH + (addBias ? 1 : 0))];
  int weightsDim = filterW * filterH + (addBias ? 1 : 0);
  int gpo0 = gp0 * groupSize_k0_M;
  int gpo1 = gp1 * groupSize_k0_N;

  for (int i = 0; i < batchSize; i++) {
    int batchOffsetIn = i * numOfInputFeatureMaps * inputFeatureMapH * inputFeatureMapW;
    int batchOffsetOut = i * (numOfOutputFeatureMaps * outputFeatureMapH * outputFeatureMapW);
    int featureMapOffsetOut = batchOffsetOut + gb1 * outputFeatureMapH * outputFeatureMapW;
    int rowOut = gb0 / outputFeatureMapW;
    int colOut = gb0 % outputFeatureMapW;
    float out = 0;
    if (gb1 < numOfOutputFeatureMaps && lo1 < groupSize_k0_N ) {
      for(int c0 = lo0; c0 < weightsDim; c0 += groupSize_k0_M) {
        shared_W[lo1 * weightsDim + c0] = weights[gb1 * weightsDim + c0];
      }
    }
    if (gb0 < inputFeatureMapW && gb1 < inputFeatureMapH) {
      for(int c0 = lo0; c0 < inputFeatureMapW; c0 += groupSize_k0_M) {
        for(int c1 = lo1; c1 < inputFeatureMapH; c1 += groupSize_k0_N) {
          for (int k = 0; k < numOfInputFeatureMaps; k++) {
            int featureMapOffsetIn = batchOffsetIn + k * inputFeatureMapH * inputFeatureMapW;
            shared_I[c1][c0] += inputFeatureMaps[featureMapOffsetIn + c1 * inputFeatureMapW + c0];
          }
        }
      }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    

      for (int l = 0; l < weightsDim && l < filterW * filterH; l++) {
        int rowIn = l / filterW + rowOut * stride;
        int colIn = l % filterW + colOut * stride;
        out += 
            inputFeatureMaps[featureMapOffsetIn + rowIn * inputFeatureMapW + colIn] * shared_W[lo1 * weightsDim + l];
      }
      if (addBias == 1 && (filterW * filterH <= weightsDim)) {
        out += shared_W[lo1 * weightsDim + weightsDim - 1];
      }
    
    out = activationFunction(out);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    outputFeatureMaps[featureMapOffsetOut + rowOut * outputFeatureMapW + colOut] = out;
  }
}*/
 