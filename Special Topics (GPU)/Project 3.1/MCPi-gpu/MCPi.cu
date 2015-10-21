#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 768

int getNeededBlocks(int total_threads);
void saveArray(int *array, int total_darts);
void initFrequencyArray(int *array, int size);
__global__ void initRandom(unsigned int seed, curandState_t* states);
__global__ void calculatePoint(int* result, curandState_t* states, int total_darts, int* values);

int main(int argc, char *argv[]) {
  if(argc != 2) {
    printf("Error: You must provide the number of tests to perform\n");
  }
  else {
    //Obtain the total darts to throw from the terminal
    int total_darts = atoi(argv[1]);
    int result = 0;
    int* values;

    int* dev_result;
    int* dev_values;

    if(total_darts >= 0) {
      //Keep track of seed value for every thread
      curandState_t* states;
      //Allocate space on GPU for random states, randomly generated values and the result
      cudaMalloc((void**) &states, sizeof(curandState_t) * total_darts);
      cudaMalloc((void**) &dev_values, sizeof(int) * 10);
      cudaMalloc((void**) &dev_result, sizeof(int));

      //Obtain the number of blocks that are going to be needed for execution
      int neededBlocks = getNeededBlocks(total_darts);
      //Initialize the values array with 0s
      values = (int*) malloc(sizeof(int) * 10);
      initFrequencyArray(values, 10);
      //Initialize the dev_result variable to 0 by copying the host variable result
      cudaMemcpy(dev_result, &result, sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(dev_values, values, sizeof(int) * 10, cudaMemcpyHostToDevice);

      //Initialize all of the random states on the GPU
      initRandom<<<neededBlocks, BLOCK_SIZE>>>(time(NULL), states);
      cudaThreadSynchronize();
      //Perform the computation of random numbers and count them if they belong to the circle
      calculatePoint<<<neededBlocks, BLOCK_SIZE>>>(dev_result, states, total_darts, dev_values);
      cudaThreadSynchronize();

      //Copy the obtained dev_result and dev_values from device back to host
      cudaMemcpy(&result, dev_result, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(values, dev_values, sizeof(int) * 10, cudaMemcpyDeviceToHost);
      //Obtain the value of pi using the count of points inside the circle
      float pi = (float) ((float)result*4)/ (float) total_darts;
      printf("Estimated value of pi = %2.9f\n", pi);
      //Save the random values histogram to a file
      saveArray(values, total_darts);
      //Free the previously allocated memory, because we care
      cudaFree(states);
      cudaFree(dev_result);
      cudaFree(dev_values);
      free(values);
    } else {
        printf("Error: The number of tests provided must be a positive number\n");
    }
  }
  //Graceful
  exit(0);
}

void saveArray(int *array, int total_darts) {
  int i;
  FILE *file;
  //Open the freq.dat file
  file = fopen("freq.dat","w");
  if(file==NULL){
    //There was an error opening the file
    printf("Error: The file freq.dat could not be opened\n");
  }
  else {
    int total = total_darts * 2;
    //For each of the values between 0.0 and 0.9
    for(i = 0; i < 10; i++) {
      //Store the current value and the frequency of it sepparated by a comma
      fprintf(file, "%2.1f,%2.4f\n", (float)i/10, (float)array[i]/total);
    }
    fclose(file);
  }
}

int getNeededBlocks(int total_darts) {
  //Obtain the total amount of required blocks by dividing the number of tests by the block's size
  int result = ceil((float)total_darts/BLOCK_SIZE);
  return result;
}

void initFrequencyArray(int* array, int size) {
  //Initialize the desired array with 0s
  int i;
  for(i = 0; i < size; i++){
    array[i] = 0;
  }
}

__global__ void calculatePoint(int* result, curandState_t* states, int total_darts, int* values){
  float x, y;
  //Obtain the global id of a thread in order to know if it should be counted for the test
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int globalId = row*blockDim.x + col;
  if(globalId < total_darts) {
    //Calculate random values for x and y
    x = curand_uniform(&states[threadIdx.x]);
    y = curand_uniform(&states[threadIdx.x]);
    if((x*x + y*y) <= 1) {
      //The point is inside the circle so the partial sum is increased
      atomicAdd(result, 1);
    }
    //Access the position in the histogram where x and y are located and increase them by 1
    atomicAdd(&values[(int)floor(x*10)], 1);
    atomicAdd(&values[(int)floor(y*10)], 1);
  }
}

__global__ void initRandom(unsigned int seed, curandState_t* states) {
  //Initialize the currand using the desired
  curand_init(seed, threadIdx.x, 0, &states[threadIdx.x]);
}
