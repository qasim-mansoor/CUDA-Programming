#include <stdio.h>
#define N 8
#define numThread 2 // 2 threads in a block
#define numBlock 4  // 4 blocks

__global__ void add( int *a, int *b, int *c ) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    while (tid < N) {
        c[tid] = a[tid] + b[tid];       
        tid += blockDim.x;       
    }
}



int main( void ) {
    int *a, *b, *c;               
    int *dev_a, *dev_b, *dev_c;   

    // Allocate the memory on the CPU
    a = (int*)malloc( N * sizeof(int) );
    b = (int*)malloc( N * sizeof(int) );
    c = (int*)malloc( N * sizeof(int) );

    // Fill a and b with dummy values
    for (int i=0; i<N; i++) {
        a[i] = i;
        b[i] = i;
    }

    // Allocate memory on GPU
     cudaMalloc( (void**)&dev_a, N * sizeof(int) );
     cudaMalloc( (void**)&dev_b, N * sizeof(int) );
     cudaMalloc( (void**)&dev_c, N * sizeof(int) );

    // Copy arrays a and b to the GPU
     cudaMemcpy( dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice );
     cudaMemcpy( dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice );

    add<<<numBlock,numThread>>>( dev_a, dev_b, dev_c );
    
    // Copy array c from GPU back to CPU
    cudaMemcpy( c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost );

    for (int i=0; i<N; i++) {
        printf( "%d + %d = %d\n", a[i], b[i], c[i] );
    }

    // Freeing the memory on the GPU
    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( dev_c );

    return 0;
}