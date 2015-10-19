#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float estimatePi(int total_points);

int main(int argc, char *argv[]) {
  if(argc != 2) {
    printf("Error: You must provide the length of the columns/rows\n");
  }
  else {
    int total_darts = atoi(argv[1]);
    if(total_darts >= 0) {
      float pi = estimatePi(total_darts);
      printf("Estimated value of pi = %2.10f\n",pi);
    } else {
        printf("Error: The size provided must be a positive number\n");
    }
  }
  //Graceful
  exit(0);
}

float estimatePi(int total_points) {
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
  }
  return (float) ((float)in_circle*4)/ (float) total_points;
}
