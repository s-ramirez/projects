#include <stdio.h>
#include <stdlib.h>

void initMatrix(int size, float matrix[size][size]);
void printMatrix(int size, float matrix[size][size]);
void multiplyMatrices(int size, float matrix1[size][size], float matrix2[size][size]);

int main(int argc, char *argv[]) {
  if(argc != 2) {
    printf("Error: You must provide the length of the columns/rows\n");
  }
  else {
    //TODO: Revisar si el numero es positivo y redondear si no es entero
    int size = atoi(argv[1]);
    float matrix1[size][size], matrix2[size][size];
    initMatrix(size, matrix1);
    initMatrix(size, matrix2);
    multiplyMatrices(size, matrix1, matrix2);
    printMatrix(size, matrix1);
  }
  exit(0);
}

void initMatrix(int size, float matrix[size][size]) {
  int i, j;
  srand48(time(NULL));
  for(j = 0; j < size; j++) {
    for(i = 0; i < size; i++) {
      matrix[i][j] = (float) rand();
      matrix[i][j] += drand48();
    }
  }
}

void printMatrix(int size, float matrix[size][size]) {
  int i,j;
  for(j = 0; j < size; j++) {
    for(i = 0; i < size; i++) {
      printf("%9.2f\t",matrix[i][j]);
    }
    printf("\n");
  }
}

void multiplyMatrices(int size, float matrix1[size][size], float matrix2[size][size]) {

}
