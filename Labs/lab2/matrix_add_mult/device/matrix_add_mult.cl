__kernel void MultAdd(
    __global float *outputD,
    int sizeM,
    int sizeN,
    int sizeP,
    __global float *inputA,
    __global float *inputB,
    __global float *inputC) {
        int row = get_global_id(1);
        int col = get_global_id(0);

        float sum = 0.0f;

        for (int i = 0; i < sizeN; i++) {
            sum += inputA[row * sizeN + i] * inputB[i * sizeN + col]; 
        }
        sum += inputC[row * sizeM + col];
        outputD[row * sizeM + col] = sum;
        
    }