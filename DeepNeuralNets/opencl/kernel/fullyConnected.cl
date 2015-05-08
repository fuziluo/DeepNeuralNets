__kernel void weightedSumSigmoid(__global float *preActivations, __global float *weights, __global float *activations, int batchSize, int numOfPerceptrons, int weightsDim, int activationDim)
{
    int gid0 = get_group_id(0), gid1 = get_group_id(1);
    int lid0 = get_local_id(0), lid1 = get_local_id(1);
    __local float shared_A[32][32];
    float private_C[2][2];
    __local float shared_B[32][32];
    int N = activationDim;

    // #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    for (int i = 32 * gid0; i < batchSize; i += 8192)
      for (int j = 32 * gid1; j < numOfPerceptrons; j += 8192) {
        for (int k = 0; k <= (weightsDim >= 1 ? weightsDim - 1 : 0); k += 32) {
          for (int c0 = lid0; c0 <= min(31, weightsDim - k - 1); c0 += 16)
            for (int c1 = lid1; c1 <= min(31, numOfPerceptrons - j - 1); c1 += 16)
              shared_B[c0][c1] = weights[(j + c1) * weightsDim + (k + c0)];
          for (int c0 = lid0; c0 <= min(31, batchSize - i - 1); c0 += 16)
            for (int c1 = lid1; c1 <= min(31, weightsDim - k - 1); c1 += 16)
              shared_A[c0][c1] = preActivations[(i + c0) * weightsDim + (k + c1)];
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          if (k == 0 && batchSize >= lid0 + i + 1 && numOfPerceptrons >= lid1 + j + 1) {
            private_C[0][0] = 0;
            if (numOfPerceptrons >= lid1 + j + 17)
              private_C[0][1] = 0;
            if (batchSize >= lid0 + i + 17) {
              private_C[1][0] = 0;
              if (numOfPerceptrons >= lid1 + j + 17)
                private_C[1][1] = 0;
            }
          }
          if (batchSize >= lid0 + i + 1 && numOfPerceptrons >= lid1 + j + 1)
            for (int c2 = 0; c2 <= min(31, weightsDim - k - 1); c2 += 1) {
              private_C[0][0] += shared_A[lid0][c2] * shared_B[c2][lid1];
              if (numOfPerceptrons >= lid1 + j + 17)
                private_C[0][1] += shared_A[lid0][c2] * shared_B[c2][lid1 + 16];
              if (batchSize >= lid0 + i + 17) {
                private_C[1][0] += shared_A[lid0 + 16][c2] * shared_B[c2][lid1];
                if (numOfPerceptrons >= lid1 + j + 17)
                  private_C[1][1] += shared_A[lid0 + 16][c2] * shared_B[c2][lid1 + 16];
              }
            }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
        if (numOfPerceptrons >= j + 1 && batchSize >= i + 1)
          if (numOfPerceptrons >= lid1 + j + 1 && numOfPerceptrons >= lid1 + j + 1 && batchSize >= lid0 + i + 1 && batchSize >= lid0 + i + 1) {
            activations[(lid0 + i) * N + (lid1 + j)] = 1.0f / (1 + exp(-private_C[0][0]));
            if (numOfPerceptrons >= lid1 + j + 17 && numOfPerceptrons >= lid1 + j + 17)
              activations[(lid0 + i) * N + (lid1 + j + 16)] = 1.0f / (1 + exp(-private_C[0][1]));
            if (batchSize >= lid0 + i + 17 && batchSize >= lid0 + i + 17) {
              activations[(lid0 + i + 16) * N + (lid1 + j)] = 1.0f / (1 + exp(-private_C[1][0]));
              if (numOfPerceptrons >= lid1 + j + 17 && numOfPerceptrons >= lid1 + j + 17)
                activations[(lid0 + i + 16) * N + (lid1 + j + 16)] = 1.0f / (1 + exp(-private_C[1][1]));
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
    __local float shared_A[32][32];
    __local float shared_B[32][32];
    float private_C[2][2];
    __local float shared_Act[32][32];
    // int N = nextWeightsDim;

    #define min(x,y)    ((x) < (y) ? (x) : (y))
    for (int i = 32 * gid0; i < batchSize; i += 8192)
      for (int j = 32 * gid1; j < numOfPerceptrons; j += 8192) {
        // if (numOfPerceptrons >= j + 1)
        for (int c0 = lid0; c0 <= min(31, batchSize - i - 1); c0 += 16)
          for (int c1 = lid1; c1 <= min(31, numOfPerceptrons - j - 1); c1 += 16)
            shared_Act[c0][c1] = activations[(i + c0) * nextWeightsDim + (j + c1)];
        for (int k = 0; k <= (nextNumOfNodes >= 1 ? nextNumOfNodes - 1 : 0); k += 32) {
          for (int c0 = lid0; c0 <= min(31, nextNumOfNodes - k - 1); c0 += 16)
            for (int c1 = lid1; c1 <= min(31, numOfPerceptrons - j - 1); c1 += 16)
              shared_B[c0][c1] = nextWeights[(k + c0) * nextWeightsDim + (j + c1)];
          for (int c0 = lid0; c0 <= min(31, batchSize - i - 1); c0 += 16)
            for (int c1 = lid1; c1 <= min(31, nextNumOfNodes - k - 1); c1 += 16)
              shared_A[c0][c1] = nextError[(i + c0) * nextNumOfNodes + (k + c1)];
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          if (k == 0 && batchSize >= lid0 + i + 1 && numOfPerceptrons >= lid1 + j + 1) {
            private_C[0][0] = 0;
            if (numOfPerceptrons >= lid1 + j + 17)
              private_C[0][1] = 0;
            if (batchSize >= lid0 + i + 17) {
              private_C[1][0] = 0;
              if (numOfPerceptrons >= lid1 + j + 17)
                private_C[1][1] = 0;
            }
          }
          if (batchSize >= lid0 + i + 1 && numOfPerceptrons >= lid1 + j + 1)
            for (int c2 = 0; c2 <= min(31, nextNumOfNodes - k - 1); c2 += 1) {
              private_C[0][0] += (shared_A[lid0][c2] * shared_B[c2][lid1]) * shared_Act[lid0][lid1] * (1 - shared_Act[lid0][lid1]);
              if (numOfPerceptrons >= lid1 + j + 17)
                private_C[0][1] += (shared_A[lid0][c2] * shared_B[c2][lid1 + 16]) * shared_Act[lid0][lid1 + 16] * (1 - shared_Act[lid0][lid1 + 16]);
              if (batchSize >= lid0 + i + 17) {
                private_C[1][0] += (shared_A[lid0 + 16][c2] * shared_B[c2][lid1]) * shared_Act[lid0 + 16][lid1] * (1 - shared_Act[lid0 + 16][lid1]);
                if (numOfPerceptrons >= lid1 + j + 17)
                  private_C[1][1] += (shared_A[lid0 + 16][c2] * shared_B[c2][lid1 + 16]) * shared_Act[lid0 + 16][lid1 + 16] * (1 - shared_Act[lid0 + 16][lid1 + 16]);
              }
            }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
        if (numOfPerceptrons >= j + 1 && batchSize >= i + 1)
          if (numOfPerceptrons >= lid1 + j + 1 && numOfPerceptrons >= lid1 + j + 1 && batchSize >= lid0 + i + 1 && batchSize >= lid0 + i + 1) {
            error[(lid0 + i) * numOfPerceptrons + (lid1 + j)] = private_C[0][0];
            if (numOfPerceptrons >= lid1 + j + 17 && numOfPerceptrons >= lid1 + j + 17)
              error[(lid0 + i) * numOfPerceptrons + (lid1 + j + 16)] = private_C[0][1];
            if (batchSize >= lid0 + i + 17 && batchSize >= lid0 + i + 17) {
              error[(lid0 + i + 16) * numOfPerceptrons + (lid1 + j)] = private_C[1][0];
              if (numOfPerceptrons >= lid1 + j + 17 && numOfPerceptrons >= lid1 + j + 17)
                error[(lid0 + i + 16) * numOfPerceptrons + (lid1 + j + 16)] = private_C[1][1];
            }
          }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      }
}

__kernel void weightedSumBackPropSigmoidUpdateGradients(__global float *error, __global float *preActivations, __global float *gradients, int numOfPerceptrons, int weightsDim, int batchSize)
{
    int gid0 = get_group_id(0), gid1 = get_group_id(1);
    int lid0 = get_local_id(0), lid1 = get_local_id(1);
    __local float shared_A[32][32];
    float private_C[2][2];
    __local float shared_B[32][32];
    // int N = activationDim;

    // #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    for (int i = 32 * gid0; i < numOfPerceptrons; i += 8192)
      for (int j = 32 * gid1; j < weightsDim; j += 8192) {
        for (int k = 0; k <= (batchSize >= 1 ? batchSize - 1 : 0); k += 32) {
          for (int c0 = lid0; c0 <= min(31, numOfPerceptrons - i - 1); c0 += 16)
            for (int c1 = lid1; c1 <= min(31, batchSize - k - 1); c1 += 16)
              shared_A[c0][c1] = error[(i + c0) + (k + c1) * numOfPerceptrons];
          for (int c0 = lid1; c0 <= min(31, weightsDim - j - 1); c0 += 16)
            for (int c1 = lid0; c1 <= min(31, batchSize - k - 1); c1 += 16)
              shared_B[c1][c0] = preActivations[(j + c0) + (k + c1) * weightsDim];
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
          if (k == 0 && numOfPerceptrons >= lid0 + i + 1 && weightsDim >= lid1 + j + 1) {
            private_C[0][0] = 0;
            if (weightsDim >= lid1 + j + 17)
              private_C[0][1] = 0;
            if (numOfPerceptrons >= lid0 + i + 17) {
              private_C[1][0] = 0;
              if (weightsDim >= lid1 + j + 17)
                private_C[1][1] = 0;
            }
          }
          if (numOfPerceptrons >= lid0 + i + 1 && weightsDim >= lid1 + j + 1)
            for (int c2 = 0; c2 <= min(31, batchSize - k - 1); c2 += 1) {
              private_C[0][0] += (shared_A[lid0][c2] * shared_B[c2][lid1]);
              if (weightsDim >= lid1 + j + 17)
                private_C[0][1] += (shared_A[lid0][c2] * shared_B[c2][lid1 + 16]);
              if (numOfPerceptrons >= lid0 + i + 17) {
                private_C[1][0] += (shared_A[lid0 + 16][c2] * shared_B[c2][lid1]);
                if (weightsDim >= lid1 + j + 17)
                  private_C[1][1] += (shared_A[lid0 + 16][c2] * shared_B[c2][lid1 + 16]);
              }
            }
          barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
        if (weightsDim >= j + 1 && numOfPerceptrons >= i + 1)
          if (weightsDim >= lid1 + j + 1 && weightsDim >= lid1 + j + 1 && numOfPerceptrons >= lid0 + i + 1 && numOfPerceptrons >= lid0 + i + 1) {
            gradients[(lid0 + i) * weightsDim + (lid1 + j)] = private_C[0][0];
            if (weightsDim >= lid1 + j + 17 && weightsDim >= lid1 + j + 17)
              gradients[(lid0 + i) * weightsDim + (lid1 + j + 16)] = private_C[0][1];
            if (numOfPerceptrons >= lid0 + i + 17 && numOfPerceptrons >= lid0 + i + 17) {
              gradients[(lid0 + i + 16) * weightsDim + (lid1 + j)] = private_C[1][0];
              if (weightsDim >= lid1 + j + 17 && weightsDim >= lid1 + j + 17)
                gradients[(lid0 + i + 16) * weightsDim + (lid1 + j + 16)] = private_C[1][1];
            }
          }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      }
}


