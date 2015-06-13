//  #define groupSize 8

__kernel void weightedSumSigmoid(__global float *preActivations, __global float *weights, __global float *activations, int batchSize, int numOfPerceptrons, int weightsDim, int activationDim)
{
    int gid0 = get_group_id(0), gid1 = get_group_id(1);
    int lid0 = get_local_id(0), lid1 = get_local_id(1);
    __local float shared_A[2 * groupSize_k0_M][groupSize_k0_K];
    float private_C[2][2];
    __local float shared_B[groupSize_k0_K][2 * groupSize_k0_N];
    int N = activationDim;

    #define min(x,y)    ((x) < (y) ? (x) : (y))
    for (int i = 2 * groupSize_k0_M * gid0; i < batchSize; i += 8192)
      for (int j = 2 * groupSize_k0_N * gid1; j < numOfPerceptrons; j += 8192) {
        for (int k = 0; k <= (weightsDim >= 1 ? weightsDim - 1 : 0); k += groupSize_k0_K) {
          for (int c0 = lid0; c0 <= min(groupSize_k0_K - 1, weightsDim - k - 1); c0 += groupSize_k0_M)
            for (int c1 = lid1; c1 <= min(2 * groupSize_k0_N - 1, numOfPerceptrons - j - 1); c1 += groupSize_k0_N)
              shared_B[c0][c1] = weights[(j + c1) * weightsDim + (k + c0)];
          for (int c0 = lid0; c0 <= min(2 * groupSize_k0_M - 1, batchSize - i - 1); c0 += groupSize_k0_M)
            for (int c1 = lid1; c1 <= min(groupSize_k0_K - 1, weightsDim - k - 1); c1 += groupSize_k0_N)
              shared_A[c0][c1] = preActivations[(i + c0) * weightsDim + (k + c1)];
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          if (k == 0 && batchSize >= lid0 + i + 1 && numOfPerceptrons >= lid1 + j + 1) {
            private_C[0][0] = 0;
            if (numOfPerceptrons >= lid1 + j + groupSize_k0_N + 1)
              private_C[0][1] = 0;
            if (batchSize >= lid0 + i + groupSize_k0_M + 1) {
              private_C[1][0] = 0;
              if (numOfPerceptrons >= lid1 + j + groupSize_k0_N + 1)
                private_C[1][1] = 0;
            }
          }
          if (batchSize >= lid0 + i + 1 && numOfPerceptrons >= lid1 + j + 1)
            for (int c2 = 0; c2 <= min(groupSize_k0_K - 1, weightsDim - k - 1); c2 += 1) {
              private_C[0][0] += shared_A[lid0][c2] * shared_B[c2][lid1];
              if (numOfPerceptrons >= lid1 + j + groupSize_k0_N + 1)
                private_C[0][1] += shared_A[lid0][c2] * shared_B[c2][lid1 + groupSize_k0_N];
              if (batchSize >= lid0 + i + groupSize_k0_M + 1) {
                private_C[1][0] += shared_A[lid0 + groupSize_k0_M][c2] * shared_B[c2][lid1];
                if (numOfPerceptrons >= lid1 + j + groupSize_k0_N + 1)
                  private_C[1][1] += shared_A[lid0 + groupSize_k0_M][c2] * shared_B[c2][lid1 + groupSize_k0_N];
              }
            }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
          if (numOfPerceptrons >= lid1 + j + 1 && batchSize >= lid0 + i + 1) {
            activations[(lid0 + i) * N + (lid1 + j)] = 1.0f / (1 + exp(-private_C[0][0]));
            if (numOfPerceptrons == lid1 + j + 1 && N > numOfPerceptrons)
              activations[(lid0 + i) * N + (lid1 + j + 1)] = 1.0f;
            if (numOfPerceptrons >= lid1 + j + groupSize_k0_N + 1) {
              activations[(lid0 + i) * N + (lid1 + j + groupSize_k0_N)] = 1.0f / (1 + exp(-private_C[0][1]));
              if (numOfPerceptrons == lid1 + j + groupSize_k0_N + 1 && N > numOfPerceptrons)
                activations[(lid0 + i) * N + (lid1 + j + groupSize_k0_N + 1)] = 1.0f;
            }
            if (batchSize >= lid0 + i + groupSize_k0_M + 1) {
              activations[(lid0 + i + groupSize_k0_M) * N + (lid1 + j)] = 1.0f / (1 + exp(-private_C[1][0]));
              if (numOfPerceptrons == lid1 + j + 1 && N > numOfPerceptrons)
                activations[(lid0 + i + groupSize_k0_M) * N + (lid1 + j + 1)] = 1.0f;
              if (numOfPerceptrons >= lid1 + j + groupSize_k0_N + 1){
                activations[(lid0 + i + groupSize_k0_M) * N + (lid1 + j + groupSize_k0_N)] = 1.0f / (1 + exp(-private_C[1][1]));
                if (numOfPerceptrons == lid1 + j + groupSize_k0_N + 1 && N > numOfPerceptrons)
                  activations[(lid0 + i + groupSize_k0_M) * N + (lid1 + j + groupSize_k0_N + 1)] = 1.0f;
              }
            }
          } 
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      }
}

__kernel void weightedSumBackPropSigmoidCalcErr(__global float *nextError, __global float *nextWeights, __global float *error, 
                                          int batchSize, int numOfPerceptrons, int nextNumOfNodes, int nextWeightsDim, __global float *activations)
{
    int gid0 = get_group_id(0), gid1 = get_group_id(1);
    int lid0 = get_local_id(0), lid1 = get_local_id(1);
    __local float shared_A[2 * groupSize_k1_M][groupSize_k1_K];
    __local float shared_B[groupSize_k1_K][2 * groupSize_k1_N];
    float private_C[2][2];
    __local float shared_Act[2 * groupSize_k1_M][2 * groupSize_k1_N];
    // int N = nextWeightsDim;

    #define min(x,y)    ((x) < (y) ? (x) : (y))
    for (int i = 2 * groupSize_k1_M * gid0; i < batchSize; i += 8192)
      for (int j = 2 * groupSize_k1_N * gid1; j < numOfPerceptrons; j += 8192) {
        // if (numOfPerceptrons >= j + 1)
        for (int c0 = lid0; c0 <= min(2 * groupSize_k1_M - 1, batchSize - i - 1); c0 += groupSize_k1_M)
          for (int c1 = lid1; c1 <= min(2 * groupSize_k1_N - 1, numOfPerceptrons - j - 1); c1 += groupSize_k1_N)
            shared_Act[c0][c1] = activations[(i + c0) * nextWeightsDim + (j + c1)];
        for (int k = 0; k <= (nextNumOfNodes >= 1 ? nextNumOfNodes - 1 : 0); k += groupSize_k1_K) {
          for (int c0 = lid0; c0 <= min(groupSize_k1_K - 1, nextNumOfNodes - k - 1); c0 += groupSize_k1_M)
            for (int c1 = lid1; c1 <= min(2 * groupSize_k1_N - 1, numOfPerceptrons - j - 1); c1 += groupSize_k1_N)
              shared_B[c0][c1] = nextWeights[(k + c0) * nextWeightsDim + (j + c1)];
          for (int c0 = lid0; c0 <= min(2 * groupSize_k1_M - 1, batchSize - i - 1); c0 += groupSize_k1_M)
            for (int c1 = lid1; c1 <= min(groupSize_k1_K - 1, nextNumOfNodes - k - 1); c1 += groupSize_k1_N)
              shared_A[c0][c1] = nextError[(i + c0) * nextNumOfNodes + (k + c1)];
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          if (k == 0 && batchSize >= lid0 + i + 1 && numOfPerceptrons >= lid1 + j + 1) {
            private_C[0][0] = 0;
            if (numOfPerceptrons >= lid1 + j + groupSize_k1_N + 1)
              private_C[0][1] = 0;
            if (batchSize >= lid0 + i + groupSize_k1_M + 1) {
              private_C[1][0] = 0;
              if (numOfPerceptrons >= lid1 + j + groupSize_k1_N + 1)
                private_C[1][1] = 0;
            }
          }
          if (batchSize >= lid0 + i + 1 && numOfPerceptrons >= lid1 + j + 1)
            for (int c2 = 0; c2 <= min(groupSize_k1_K - 1, nextNumOfNodes - k - 1); c2 += 1) {
              private_C[0][0] += (shared_A[lid0][c2] * shared_B[c2][lid1]) * shared_Act[lid0][lid1] * (1 - shared_Act[lid0][lid1]);
              if (numOfPerceptrons >= lid1 + j + groupSize_k1_N + 1)
                private_C[0][1] += (shared_A[lid0][c2] * shared_B[c2][lid1 + groupSize_k1_N]) * shared_Act[lid0][lid1 + groupSize_k1_N] * (1 - shared_Act[lid0][lid1 + groupSize_k1_N]);
              if (batchSize >= lid0 + i + groupSize_k1_M + 1) {
                private_C[1][0] += (shared_A[lid0 + groupSize_k1_M][c2] * shared_B[c2][lid1]) * shared_Act[lid0 + groupSize_k1_M][lid1] * (1 - shared_Act[lid0 + groupSize_k1_M][lid1]);
                if (numOfPerceptrons >= lid1 + j + groupSize_k1_N + 1)
                  private_C[1][1] += (shared_A[lid0 + groupSize_k1_M][c2] * shared_B[c2][lid1 + groupSize_k1_N]) * shared_Act[lid0 + groupSize_k1_M][lid1 + groupSize_k1_N] * (1 - shared_Act[lid0 + groupSize_k1_M][lid1 + groupSize_k1_N]);
              }
            }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
        if (numOfPerceptrons >= lid1 + j + 1&& batchSize >= lid0 + i + 1) {
          error[(lid0 + i) * numOfPerceptrons + (lid1 + j)] = private_C[0][0];
          if (numOfPerceptrons >= lid1 + j + groupSize_k1_N + 1)
            error[(lid0 + i) * numOfPerceptrons + (lid1 + j + groupSize_k1_N)] = private_C[0][1];
          if (batchSize >= lid0 + i + groupSize_k1_M + 1) {
            error[(lid0 + i + groupSize_k1_M) * numOfPerceptrons + (lid1 + j)] = private_C[1][0];
            if (numOfPerceptrons >= lid1 + j + groupSize_k1_N + 1)
              error[(lid0 + i + groupSize_k1_M) * numOfPerceptrons + (lid1 + j + groupSize_k1_N)] = private_C[1][1];
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      }
}

__kernel void weightedSumBackPropSigmoidUpdateGradients(__global float *error, __global float *preActivations, __global float *gradients, int numOfPerceptrons, int weightsDim, int batchSize)
{
    int gid0 = get_group_id(0), gid1 = get_group_id(1);
    int lid0 = get_local_id(0), lid1 = get_local_id(1);
    __local float shared_A[2 * groupSize_k2_M][groupSize_k2_K];
    float private_C[2][2];
    __local float shared_B[groupSize_k2_K][2 * groupSize_k2_N];
    // int N = activationDim;

    #define min(x,y)    ((x) < (y) ? (x) : (y))
    for (int i = 2 * groupSize_k2_M * gid0; i < numOfPerceptrons; i += 8192)
      for (int j = 2 * groupSize_k2_N * gid1; j < weightsDim; j += 8192) {
        for (int k = 0; k <= (batchSize >= 1 ? batchSize - 1 : 0); k += groupSize_k2_K) {
          for (int c0 = lid0; c0 <= min(groupSize_k2_K - 1, batchSize - k - 1); c0 += groupSize_k2_M)
            for (int c1 = lid1; c1 <= min(2 * groupSize_k2_N - 1, weightsDim - j - 1); c1 += groupSize_k2_N)
              shared_B[c0][c1] = preActivations[(j + c1) + (k + c0) * weightsDim];
          for (int c0 = lid0; c0 <= min(2 * groupSize_k2_M - 1, numOfPerceptrons - i - 1); c0 += groupSize_k2_M)
            for (int c1 = lid1; c1 <= min(groupSize_k2_K - 1, batchSize - k - 1); c1 += groupSize_k2_N)
              shared_A[c0][c1] = error[(i + c0) + (k + c1) * numOfPerceptrons];
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          if (k == 0 && numOfPerceptrons >= lid0 + i + 1 && weightsDim >= lid1 + j + 1) {
            private_C[0][0] = 0;
            if (weightsDim >= lid1 + j + groupSize_k2_N + 1)
              private_C[0][1] = 0;
            if (numOfPerceptrons >= lid0 + i + groupSize_k2_M + 1) {
              private_C[1][0] = 0;
              if (weightsDim >= lid1 + j + groupSize_k2_N + 1)
                private_C[1][1] = 0;
            }
          }
          if (numOfPerceptrons >= lid0 + i + 1 && weightsDim >= lid1 + j + 1)
            for (int c2 = 0; c2 <= min(groupSize_k2_K - 1, batchSize - k - 1); c2 += 1) {
              private_C[0][0] += (shared_A[lid0][c2] * shared_B[c2][lid1]);
              if (weightsDim >= lid1 + j + groupSize_k2_N + 1)
                private_C[0][1] += (shared_A[lid0][c2] * shared_B[c2][lid1 + groupSize_k2_N]);
              if (numOfPerceptrons >= lid0 + i + groupSize_k2_M + 1) {
                private_C[1][0] += (shared_A[lid0 + groupSize_k2_M][c2] * shared_B[c2][lid1]);
                if (weightsDim >= lid1 + j + groupSize_k2_N + 1)
                  private_C[1][1] += (shared_A[lid0 + groupSize_k2_M][c2] * shared_B[c2][lid1 + groupSize_k2_N]);
              }
            }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
        if (weightsDim >= lid1 + j + 1 && numOfPerceptrons >= lid0 + i + 1) {
          gradients[(lid0 + i) * weightsDim + (lid1 + j)] = private_C[0][0] / batchSize;
          if (weightsDim >= lid1 + j + groupSize_k2_N + 1)
            gradients[(lid0 + i) * weightsDim + (lid1 + j + groupSize_k2_N)] = private_C[0][1] / batchSize;
          if (numOfPerceptrons >= lid0 + i + groupSize_k2_M + 1) {
            gradients[(lid0 + i + groupSize_k2_M) * weightsDim + (lid1 + j)] = private_C[1][0] / batchSize;
            if (weightsDim >= lid1 + j + groupSize_k2_N + 1)
              gradients[(lid0 + i + groupSize_k2_M) * weightsDim + (lid1 + j + groupSize_k2_N)] = private_C[1][1] / batchSize;
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
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

