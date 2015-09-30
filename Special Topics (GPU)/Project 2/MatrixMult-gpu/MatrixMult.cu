#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float* initMatrix(int size);
void saveMatrix(int size, float *matrix);
__global__ void multiplyMatrices(int size, float *matrix1, float *matrix2, float *result, int tile_width);

int main(int argc, char *argv[]) {
  if(argc != 3) {
    printf("Error: You must provide the length of the columns/rows followed by the tile width\n");
  }
  else {
    int size = atoi(argv[1]);
    int tile_width = atoi(argv[2]);

    if(size >= 0) {
        printf("Size: %d Tile size: %d\n", size, tile_width);
        clock_t start, stop;
        float *dev_matrix1, *dev_matrix2, *dev_results;

        //Initialize matrices
        float *matrix1 = initMatrix(size);
        float *matrix2 = initMatrix(size);

        //Initialize device matrices
        int total_size = size * size * sizeof(float);
        cudaMalloc((void **) &dev_matrix1, total_size);
        cudaMalloc((void **) &dev_matrix2, total_size);
        cudaMalloc((void **) &dev_results, total_size);

        //Copy matrices to device
        cudaMemcpy(dev_matrix1, matrix1, total_size, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_matrix2, matrix2, total_size, cudaMemcpyHostToDevice);

        dim3 dimBlock(tile_width, tile_width);
        //Casting divisions to float, otherwise the result tends to be 0
        dim3 dimGrid((int)ceil((float)size/(float)dimBlock.x), (int)ceil((float)size/(float)dimBlock.y));

        //Execute the code once so that everything is initialized before meassuring
        multiplyMatrices<<<dimGrid, dimBlock>>>(size, dev_matrix1, dev_matrix2, dev_results, tile_width);

        //Store the starting time
        start = clock();

        //Perform the matrix multiplication
        multiplyMatrices<<<dimGrid, dimBlock>>>(size, dev_matrix1, dev_matrix2, dev_results, tile_width);
        cudaThreadSynchronize();

        //Store the stopping time
        stop = clock();

        float *results = (float *) malloc(total_size);
        cudaMemcpy(results, dev_results, total_size, cudaMemcpyDeviceToHost);

        //Calculate the total elapsed time
        float elapsed_time = (float)(stop - start)/CLOCKS_PER_SEC;
        printf("Elapsed Time: %2.8fs\n", elapsed_time);

        //Save the results matrix to a file and free the memory
        saveMatrix(size, results);
        free(matrix1);
        free(matrix2);
        free(results);
        cudaFree(dev_matrix1);
        cudaFree(dev_matrix2);
        cudaFree(dev_results);
    } else {
        printf("Error: The size provided must be a positive number\n");
    }
  }
  //Graceful
  exit(0);
}

float* initMatrix(int size) {
  int i, j;
  float *matrix = (float *) malloc(sizeof(float) * size * size);

  //Initialize the random generator using the current time as a seed
  srand48(time(NULL));
  for(j = 0; j < size; j++) {
    for(i = 0; i < size; i++) {
        //Assign a random number to the matrix
        matrix[i * size + j] = (float) (rand() % 100);
        //Add some random decimals to the value
        matrix[j * size + i] += drand48();
    }
  }
  return matrix;
}

void saveMatrix(int size, float *matrix) {
  int i,j;
  FILE *file;

  file = fopen("product.dat","w");
  if(file==NULL){
      printf("Error: The file product.dat could not be opened\n");
  }
  else {
      for(j = 0; j < size; j++) {
          for(i = 0; i < size; i++) {
              fprintf(file, "%2.2f\t", matrix[j * size + i]);
          }
          fprintf(file, "\n");
      }
      fclose(file);
  }
}

__global__ void multiplyMatrices(int size, float *matrix1, float *matrix2, float* results, int tile_width) {
    int k, sum = 0;
    int col = blockIdx.x*tile_width + threadIdx.x;
    int row = blockIdx.y*tile_width + threadIdx.y;
    if(col < size && row < size) {
      for(k = 0; k < size; k++)
        sum += matrix1[row * size + k] * matrix2[k * size + col];
      results[row * size + col] = sum;
    }
}
