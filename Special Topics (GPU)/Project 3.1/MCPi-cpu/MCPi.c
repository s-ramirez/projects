#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void saveArray(int size, float *array);
float estimatePi(int total_points, float* values);

int main(int argc, char *argv[]) {
  if(argc != 2) {
    printf("Error: You must provide the length of the columns/rows\n");
  }
  else {
    float *values;
    int total_darts = atoi(argv[1]);

    if(total_darts >= 0) {
      values = (float*) malloc(sizeof(float) * total_darts * 2);
      float pi = estimatePi(total_darts, values);
      printf("Estimated value of pi = %2.10f\n",pi);
      saveArray(total_darts * 2, values);
    } else {
        printf("Error: The size provided must be a positive number\n");
    }
  }
  //Graceful
  exit(0);
}

void saveArray(int size, float *array) {
  int i;
  FILE *file;

  file = fopen("freq.dat","w");
  if(file==NULL){
    printf("Error: The file freq.dat could not be opened\n");
  }
  else {
    for(i = 0; i < size; i++) {
      fprintf(file, "%2.4f\n", array[i]);
    }
    fclose(file);
  }
}

float estimatePi(int total_points, float* values) {
  srand48(time(NULL));
  int i;
  int in_circle = 0;
  float x, y;

  for(i = 0; i < total_points; i++) {
    x = (float) (rand() % 1);
    x += drand48();
    y = (float) (rand() % 1);
    y += drand48();

    if(x*x + y*y <= 1) {
      in_circle += 1;
    }
    values[i] = x;
    values[total_points+i] = y;
  }
  return (float) ((float)in_circle*4)/ (float) total_points;
}
