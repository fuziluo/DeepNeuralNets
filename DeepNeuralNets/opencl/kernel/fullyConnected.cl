//  #define groupSize 8
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

__kernel void forwardPass(__global float *preActivations, __global float *weights, __global float *activations)//, int batchSize, int numOfPerceptrons, int weightsDim, int prevActDim)
{
    int gid0 = get_group_id(0), gid1 = get_group_id(1);
    int lid0 = get_local_id(0), lid1 = get_local_id(1);
    __local float shared_A[2 * groupSize_k0_X][groupSize_k0_Z];
    float private_C[2][2] = {0,0,0,0};
    __local float shared_B[groupSize_k0_Z][2 * groupSize_k0_Y];

    // for (int i = 2 * groupSize_k0_X * gid0; i < batchSize; i += 8192)
    //   for (int j = 2 * groupSize_k0_Y * gid1; j < numOfPerceptrons; j += 8192) {
        int i = 2 * groupSize_k0_X * gid0;
        int j = 2 * groupSize_k0_Y * gid1;
        for (int k = 0; k <= (weightsDim >= 1 ? weightsDim - 1 : 0); k += groupSize_k0_Z) {
          for (int c0 = lid0; c0 <= min(groupSize_k0_Z - 1, weightsDim - k - 1); c0 += groupSize_k0_X)
            for (int c1 = lid1; c1 <= min(2 * groupSize_k0_Y - 1, numOfPerceptrons - j - 1); c1 += groupSize_k0_Y)
              shared_B[c0][c1] = weights[(j + c1) * weightsDim + (k + c0)];
          for (int c0 = lid0; c0 <= min(2 * groupSize_k0_X - 1, batchSize - i - 1); c0 += groupSize_k0_X)
            for (int c1 = lid1; c1 <= min(groupSize_k0_Z - 1, weightsDim - k - 1); c1 += groupSize_k0_Y)
              // shared_A[c0][c1] = select(preActivations[(i + c0) * prevActDim + (k + c1)], 1.0f, weightsDim != prevActDim && k + c1 == weightsDim - 1);
              if (weightsDim != prevActDim && k + c1 == weightsDim - 1)
                shared_A[c0][c1] = 1.0f;
              else
                shared_A[c0][c1] = preActivations[(i + c0) * prevActDim + (k + c1)];
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          // if (k == 0 && batchSize >= lid0 + i + 1 && numOfPerceptrons >= lid1 + j + 1) {
          //   private_C[0][0] = 0;
          //   if (numOfPerceptrons >= lid1 + j + groupSize_k0_Y + 1)
          //     private_C[0][1] = 0;
          //   if (batchSize >= lid0 + i + groupSize_k0_X + 1) {
          //     private_C[1][0] = 0;
          //     if (numOfPerceptrons >= lid1 + j + groupSize_k0_Y + 1)
          //       private_C[1][1] = 0;
          //   }
          // }
          bool cond = batchSize >= lid0 + i + 1 && numOfPerceptrons >= lid1 + j + 1;
          // if (cond)
            for (int c2 = 0; c2 <= min(groupSize_k0_Z - 1, weightsDim - k - 1); c2 += 1) {
              private_C[0][0] = select(private_C[0][0], private_C[0][0] + shared_A[lid0][c2] * shared_B[c2][lid1], cond*1);
              private_C[0][1] = select(private_C[0][1], private_C[0][1] + shared_A[lid0][c2] * shared_B[c2][lid1 + groupSize_k0_Y], numOfPerceptrons >= lid1 + j + groupSize_k0_Y + 1 && cond);
              private_C[1][0] = select(private_C[1][0], private_C[1][0] + shared_A[lid0 + groupSize_k0_X][c2] * shared_B[c2][lid1], batchSize >= lid0 + i + groupSize_k0_X + 1 && cond);
              private_C[1][1] = select(private_C[1][1], private_C[1][1] + shared_A[lid0 + groupSize_k0_X][c2] * shared_B[c2][lid1 + groupSize_k0_Y],
                batchSize >= lid0 + i + groupSize_k0_X + 1 && numOfPerceptrons >= lid1 + j + groupSize_k0_Y + 1 && cond);
              // if (numOfPerceptrons >= lid1 + j + groupSize_k0_Y + 1)
              //   private_C[0][1] += shared_A[lid0][c2] * shared_B[c2][lid1 + groupSize_k0_Y];
            }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
        bool cond1 = numOfPerceptrons >= lid1 + j + 1 && batchSize >= lid0 + i + 1;
          if (cond1) {
            activations[(lid0 + i) * numOfPerceptrons + (lid1 + j)] = activationFunction(private_C[0][0]);
            if (numOfPerceptrons >= lid1 + j + groupSize_k0_Y + 1) {
              activations[(lid0 + i) * numOfPerceptrons + (lid1 + j + groupSize_k0_Y)] = activationFunction(private_C[0][1]);
            }
            if (batchSize >= lid0 + i + groupSize_k0_X + 1) {
              activations[(lid0 + i + groupSize_k0_X) * numOfPerceptrons + (lid1 + j)] = activationFunction(private_C[1][0]);
              if (numOfPerceptrons >= lid1 + j + groupSize_k0_Y + 1){
                activations[(lid0 + i + groupSize_k0_X) * numOfPerceptrons + (lid1 + j + groupSize_k0_Y)] = activationFunction(private_C[1][1]);
              }
            }
          } 
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      // }
}

__kernel void backCalcGradients(__global float *error, __global float *preActivations, __global float *gradients)//, int numOfPerceptrons, int weightsDim, int batchSize, int prevActDim)
{
    int gid0 = get_group_id(0), gid1 = get_group_id(1);
    int lid0 = get_local_id(0), lid1 = get_local_id(1);
    __local float shared_A[2 * groupSize_k1_X][groupSize_k1_Z];
    float private_C[2][2] = {0,0,0,0};
    __local float shared_B[groupSize_k1_Z][2 * groupSize_k1_Y];

    // for (int i = 2 * groupSize_k1_X * gid0; i < numOfPerceptrons; i += 8192)
    //   for (int j = 2 * groupSize_k1_Y * gid1; j < weightsDim; j += 8192) {
        int i = 2 * groupSize_k1_X * gid0;
        int j = 2 * groupSize_k1_Y * gid1;
        for (int k = 0; k <= (batchSize >= 1 ? batchSize - 1 : 0); k += groupSize_k1_Z) {
          for (int c0 = lid0; c0 <= min(groupSize_k1_Z - 1, batchSize - k - 1); c0 += groupSize_k1_X)
            for (int c1 = lid1; c1 <= min(2 * groupSize_k1_Y - 1, weightsDim - j - 1); c1 += groupSize_k1_Y)
              if (weightsDim != prevActDim && j + c1 == weightsDim - 1)
                shared_B[c0][c1] = 1.0f;
              else
                shared_B[c0][c1] = preActivations[(j + c1) + (k + c0) * prevActDim];
          for (int c0 = lid0; c0 <= min(2 * groupSize_k1_X - 1, numOfPerceptrons - i - 1); c0 += groupSize_k1_X)
            for (int c1 = lid1; c1 <= min(groupSize_k1_Z - 1, batchSize - k - 1); c1 += groupSize_k1_Y)
              shared_A[c0][c1] = error[(i + c0) + (k + c1) * numOfPerceptrons];
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          // if (k == 0 && numOfPerceptrons >= lid0 + i + 1 && weightsDim >= lid1 + j + 1) {
          //   private_C[0][0] = 0;
          //   if (weightsDim >= lid1 + j + groupSize_k1_Y + 1)
          //     private_C[0][1] = 0;
          //   if (numOfPerceptrons >= lid0 + i + groupSize_k1_X + 1) {
          //     private_C[1][0] = 0;
          //     if (weightsDim >= lid1 + j + groupSize_k1_Y + 1)
          //       private_C[1][1] = 0;
          //   }
          // }
          bool cond = numOfPerceptrons >= lid0 + i + 1 && weightsDim >= lid1 + j + 1;
          // if (cond)
            for (int c2 = 0; c2 <= min(groupSize_k1_Z - 1, batchSize - k - 1); c2 += 1) {
              private_C[0][0] = select(private_C[0][0], private_C[0][0] + shared_A[lid0][c2] * shared_B[c2][lid1], cond*1);
              private_C[0][1] = select(private_C[0][1], private_C[0][1] + shared_A[lid0][c2] * shared_B[c2][lid1 + groupSize_k1_Y], weightsDim >= lid1 + j + groupSize_k1_Y + 1 && cond);
              private_C[1][0] = select(private_C[1][0], private_C[1][0] + shared_A[lid0 + groupSize_k1_X][c2] * shared_B[c2][lid1], numOfPerceptrons >= lid0 + i + groupSize_k1_X + 1 && cond);
              private_C[1][1] = select(private_C[1][1], private_C[1][1] + shared_A[lid0 + groupSize_k1_X][c2] * shared_B[c2][lid1 + groupSize_k1_Y], 
                weightsDim >= lid1 + j + groupSize_k1_Y + 1 && numOfPerceptrons >= lid0 + i + groupSize_k1_X + 1 && cond);
              // private_C[0][0] += (shared_A[lid0][c2] * shared_B[c2][lid1]);
              // if (weightsDim >= lid1 + j + groupSize_k1_Y + 1)
              //   private_C[0][1] += (shared_A[lid0][c2] * shared_B[c2][lid1 + groupSize_k1_Y]);
              // if (numOfPerceptrons >= lid0 + i + groupSize_k1_X + 1) {
              //   private_C[1][0] += (shared_A[lid0 + groupSize_k1_X][c2] * shared_B[c2][lid1]);
              //   if (weightsDim >= lid1 + j + groupSize_k1_Y + 1)
              //     private_C[1][1] += (shared_A[lid0 + groupSize_k1_X][c2] * shared_B[c2][lid1 + groupSize_k1_Y]);
              // }
            }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
        if (weightsDim >= lid1 + j + 1 && numOfPerceptrons >= lid0 + i + 1) {
          gradients[(lid0 + i) * weightsDim + (lid1 + j)] = private_C[0][0];
          if (weightsDim >= lid1 + j + groupSize_k1_Y + 1)
            gradients[(lid0 + i) * weightsDim + (lid1 + j + groupSize_k1_Y)] = private_C[0][1];
          if (numOfPerceptrons >= lid0 + i + groupSize_k1_X + 1) {
            gradients[(lid0 + i + groupSize_k1_X) * weightsDim + (lid1 + j)] = private_C[1][0];
            if (weightsDim >= lid1 + j + groupSize_k1_Y + 1)
              gradients[(lid0 + i + groupSize_k1_X) * weightsDim + (lid1 + j + groupSize_k1_Y)] = private_C[1][1];
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      // }
}

__kernel void backCalcPrevErr(__global float *error, __global float *weights, __global float *prevError, 
                                          /*int batchSize, int prevActDim, int numOfPerceptrons, int weightsDim, */__global float *prevActivations)
{
    int gid0 = get_group_id(0), gid1 = get_group_id(1);
    int lid0 = get_local_id(0), lid1 = get_local_id(1);
    __local float shared_A[2 * groupSize_k2_X][groupSize_k2_Z];
    __local float shared_B[groupSize_k2_Z][2 * groupSize_k2_Y];
    float private_C[2][2] = {0,0,0,0};
    // __local float shared_Act[2 * groupSize_k2_X][2 * groupSize_k2_Y];

    // for (int i = 2 * groupSize_k2_X * gid0; i < batchSize; i += 8192)
    //   for (int j = 2 * groupSize_k2_Y * gid1; j < prevActDim; j += 8192) {
        int i = 2 * groupSize_k2_X * gid0;
        int j = 2 * groupSize_k2_Y * gid1;
        for (int k = 0; k <= (numOfPerceptrons >= 1 ? numOfPerceptrons - 1 : 0); k += groupSize_k2_Z) {
          for (int c0 = lid0; c0 <= min(groupSize_k2_Z - 1, numOfPerceptrons - k - 1); c0 += groupSize_k2_X)
            for (int c1 = lid1; c1 <= min(2 * groupSize_k2_Y - 1, prevActDim - j - 1); c1 += groupSize_k2_Y)
              shared_B[c0][c1] = weights[(k + c0) * weightsDim + (j + c1)];
          for (int c0 = lid0; c0 <= min(2 * groupSize_k2_X - 1, batchSize - i - 1); c0 += groupSize_k2_X)
            for (int c1 = lid1; c1 <= min(groupSize_k2_Z - 1, numOfPerceptrons - k - 1); c1 += groupSize_k2_Y)
              shared_A[c0][c1] = error[(i + c0) * numOfPerceptrons + (k + c1)];
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          // if (k == 0 && batchSize >= lid0 + i + 1 && prevActDim >= lid1 + j + 1) {
          //   private_C[0][0] = 0;
          //   if (prevActDim >= lid1 + j + groupSize_k2_Y + 1)
          //     private_C[0][1] = 0;
          //   if (batchSize >= lid0 + i + groupSize_k2_X + 1) {
          //     private_C[1][0] = 0;
          //     if (prevActDim >= lid1 + j + groupSize_k2_Y + 1)
          //       private_C[1][1] = 0;
          //   }
          // }
          bool cond = batchSize >= lid0 + i + 1 && prevActDim >= lid1 + j + 1;
          // if (cond)
            for (int c2 = 0; c2 <= min(groupSize_k2_Z - 1, numOfPerceptrons - k - 1); c2 += 1) {
              private_C[0][0] = select(private_C[0][0], private_C[0][0] + (shared_A[lid0][c2] * shared_B[c2][lid1]), cond*1);
              private_C[0][1] = select(private_C[0][1], private_C[0][1] + (shared_A[lid0][c2] * shared_B[c2][lid1 + groupSize_k2_Y]),
                prevActDim >= lid1 + j + groupSize_k2_Y + 1 && cond*1);
              private_C[1][0] = select(private_C[1][0], private_C[1][0] + (shared_A[lid0 + groupSize_k2_X][c2] * shared_B[c2][lid1]),
                batchSize >= lid0 + i + groupSize_k2_X + 1 && cond*1);
              private_C[1][1] = select(private_C[1][1], private_C[1][1] + (shared_A[lid0 + groupSize_k2_X][c2] * shared_B[c2][lid1 + groupSize_k2_Y]),
                prevActDim >= lid1 + j + groupSize_k2_Y + 1 && batchSize >= lid0 + i + groupSize_k2_X + 1 && cond*1);
              // private_C[0][0] += (shared_A[lid0][c2] * shared_B[c2][lid1]) * derivative(shared_Act[lid0][lid1]);
              // if (prevActDim >= lid1 + j + groupSize_k2_Y + 1)
              //   private_C[0][1] += (shared_A[lid0][c2] * shared_B[c2][lid1 + groupSize_k2_Y]) * derivative(shared_Act[lid0][lid1 + groupSize_k2_Y]);
              // if (batchSize >= lid0 + i + groupSize_k2_X + 1) {
              //   private_C[1][0] += (shared_A[lid0 + groupSize_k2_X][c2] * shared_B[c2][lid1]) * derivative(shared_Act[lid0 + groupSize_k2_X][lid1]);
              //   if (prevActDim >= lid1 + j + groupSize_k2_Y + 1)
              //     private_C[1][1] += (shared_A[lid0 + groupSize_k2_X][c2] * shared_B[c2][lid1 + groupSize_k2_Y]) * derivative(shared_Act[lid0 + groupSize_k2_X][lid1 + groupSize_k2_Y]);
              // }
            }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
        if (prevActDim >= lid1 + j + 1 && batchSize >= lid0 + i + 1) {
          prevError[(lid0 + i) * prevActDim + (lid1 + j)] = private_C[0][0] * derivative(prevActivations[(lid0 + i) * prevActDim + (lid1 + j)]);//(shared_Act[lid0][lid1]);
          if (prevActDim >= lid1 + j + groupSize_k2_Y + 1)
            prevError[(lid0 + i) * prevActDim + (lid1 + j + groupSize_k2_Y)] = private_C[0][1] * derivative(prevActivations[(lid0 + i) * prevActDim + (lid1 + j + groupSize_k2_Y)]);//(shared_Act[lid0][lid1 + groupSize_k2_Y]);
          if (batchSize >= lid0 + i + groupSize_k2_X + 1) {
            prevError[(lid0 + i + groupSize_k2_X) * prevActDim + (lid1 + j)] = private_C[1][0] * derivative(prevActivations[(lid0 + i + groupSize_k2_X) * prevActDim + (lid1 + j)]);//(shared_Act[lid0][lid1 + groupSize_k2_Y]);
            if (prevActDim >= lid1 + j + groupSize_k2_Y + 1)
              prevError[(lid0 + i + groupSize_k2_X) * prevActDim + (lid1 + j + groupSize_k2_Y)] = private_C[1][1] * derivative(prevActivations[(lid0 + i + groupSize_k2_X) * prevActDim + (lid1 + j + groupSize_k2_Y)]);//(shared_Act[lid0 + groupSize_k2_X][lid1 + groupSize_k2_Y]);
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      // }
}


__kernel void updateWeights(__global float *gradients, __global float *weights, __global float *weightsUpdate, float learningRate, float momentum, float weightDecay_batchSize, int len)
{
    int b0 = get_group_id(0);
    int t0 = get_local_id(0);
    float private_weights;
    float private_weightsUpdate;

    float lr = learningRate;
    float decay = weightDecay_batchSize;

    for (int c0 = 128 * b0; c0 < len; c0 += 32768) {
      if ((c0 + t0) % weightsDim == prevActDim ) {
        lr = 2 * learningRate;
        decay = 0;
      }
      if (len >= t0 + c0 + 1) {
        private_weights = weights[t0 + c0];
        private_weightsUpdate = weightsUpdate[t0 + c0];
        private_weightsUpdate = momentum * private_weightsUpdate - lr * gradients[t0 + c0]  - lr * decay * private_weights;
        private_weights += private_weightsUpdate;
        weightsUpdate[t0 + c0] = private_weightsUpdate;
        weights[t0 + c0] = private_weights;
      }
    }
}

