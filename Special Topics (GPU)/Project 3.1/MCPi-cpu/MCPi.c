#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void saveArray(int *array, int total_darts);
void initFrequencyArray(int* array, int size);
float estimatePi(int total_points, int* values);

int main(int argc, char *argv[]) {
  if(argc != 2) {
    printf("Error: You must provide the number of tests to perform\n");
  }
  else {
    int *values;
    //Obtain the total darts to throw from the terminal
    int total_darts = atoi(argv[1]);

    if(total_darts >= 0) {
      //Initialize the values array with 0s
      values = (int*) malloc(sizeof(int) * 10);
      initFrequencyArray(values, 10);
      //Perform the computation of random numbers and count them if they belong to the circle
      float pi = estimatePi(total_darts, values);
      printf("Estimated value of pi = %2.10f\n",pi);
      //Save the random values histogram to a file
      saveArray(values, total_darts);
      //Free the previously allocated memory, because we care
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
void initFrequencyArray(int* array, int size) {
  //Initialize the desired array with 0s
  int i;
  for(i = 0; i < size; i++){
    array[i] = 0;
  }
}

float estimatePi(int total_points, int* values) {
  srand48(time(NULL));
  int i;
  int in_circle = 0;
  float x, y;

  for(i = 0; i < total_points; i++) {
    //Calculate random values for x and y
    x = (float) (rand() % 1);
    x += drand48();
    y = (float) (rand() % 1);
    y += drand48();

    if(x*x + y*y <= 1) {
      //if x^2+y^2 <= 1 this means that the random point is inside the circle
      in_circle += 1;
    }
    //Access the position in the histogram where x and y are located and increase them by 1
    values[(int)floor(x*10)] += 1;
    values[(int)floor(y*10)] += 1;
  }
  //Return the calculated value of pi
  //the number of values inside the circle * 4 divided by the total tests
  return (float) ((float)in_circle*4)/ (float) total_points;
}
