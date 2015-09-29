#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float* initMatrix(int size);
void saveMatrix(int size, float *matrix);
float* multiplyMatrices(int size, float *matrix1, float *matrix2);

int main(int argc, char *argv[]) {
  if(argc != 2) {
    printf("Error: You must provide the length of the columns/rows\n");
  }
  else {
    int size = atoi(argv[1]);
    if(size >= 0) {
        printf("Size: %d\n", size);
        clock_t start, stop;

        //Initialize matrices
        float *matrix1 = initMatrix(size);
        float *matrix2 = initMatrix(size);

        //Store the starting time
        start = clock();

        //Perform the matrix multiplication
        float *results = multiplyMatrices(size, matrix1, matrix2);

        //Store the stopping time
        stop = clock();

        //Calculate the total elapsed time
        float elapsed_time = (float)(stop - start)/CLOCKS_PER_SEC;
        printf("Elapsed Time: %2.4fs\n", elapsed_time);

        //Save the results matrix to a file and free the memory
        saveMatrix(size, results);
        free(matrix1);
        free(matrix2);
        free(results);
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

float* multiplyMatrices(int size, float *matrix1, float *matrix2) {
    int i, j, k;
    float *results = (float *) malloc(sizeof(float) * size * size);
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            float sum = 0;
            for (k = 0; k < size; k++) {
                sum += (float) matrix1[i * size + k]*matrix2[k * size + j];
            }
            results[i * size + j] = sum;
        }
    }
    return results;
}
