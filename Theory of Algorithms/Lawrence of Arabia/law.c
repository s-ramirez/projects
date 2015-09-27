#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX_DEPOTS 1000
#define INFINITY INT_MAX

typedef struct {
  int attacks;
  int depots;
  int *values;
} Dataset;

void lawrence(Dataset* inputs, int counter);

int main(int argc, char *argv[]) {
  if(argc != 2) {
    printf("Error: A file name must be provided\n");
  }
  else {
    FILE *file = fopen(argv[1], "r");

    if (file == NULL) {
      printf("Error: Could not read the file\n");
    }
    else {
      char* line = malloc(MAX_DEPOTS);
      int counter = 0;
      Dataset *inputs = (Dataset*) malloc(sizeof(Dataset)*100);

      while (fgets(line, MAX_DEPOTS, file) != NULL)  {
        Dataset newData;

        //Read and store depots
        char* token = strtok(line, " ");
        newData.depots = atoi(token);

        //Read and store number of attacks
        token = strtok(NULL, " ");
        newData.attacks = atoi(token);

        //Read and store the values
        if(newData.depots != 0 && newData.attacks != 0) {
          fgets(line, MAX_DEPOTS, file);
          newData.values = (int *)malloc(sizeof(int)*newData.depots);
          int position = 0;
          for(token = strtok(line, " "); token != NULL; token = strtok(NULL, " ")) {
            newData.values[position] = atoi(token);
            position++;
          }
          inputs[counter] = newData;
          counter++;
        }
      }
      int t = 0;
      free(line);
      fclose(file);
      lawrence(inputs, counter);
    }
  }
  exit(0);
}

void printMatrix(int size, int* matrix) {
  int i, j;
  for(j = 0; j < size; j++) {
    printf("%d|\t",j);
    for(i = 0; i < size; i++) {
      printf("%d\t", matrix[j * size + i]);
    }
    printf("\n");
  }
}

void initMatrix(int size, int* matrix, int value) {
  int i, j;
  for(i = 0; i < size; i++) {
    for(j = 0; j < size; j++) {
      matrix[j * size + i] = value;
    }
  }
}

int getMinimum(int val1, int val2) {
  if(val1 > val2) {
    return val2;
  } else {
    return val1;
  }
}

void getStrategicValues(int* values, int size, int* results) {
  int i, j, k;
  int current;
  for(j = 0; j < size; j++) {
    printf("%d ", values[j]);
    for(i = j+1; i < size; i++) {
      current = values[i];
      int sum = results[j * size + (i-1)];
      for(k = j; k < i; k++) {
        sum += current*values[k];
      }
      results[j * size + i] = sum;
    }
  }
  printf("\ninitial strategic value: %d\n\n", results[size-1]);
}

void getOptimalValues(int size, int* values, int current_bomb, int total_bombs, int* optimalValues[total_bombs]) {
  int i,j,k,bomb_pos;
  int result;
  if(current_bomb <= total_bombs) {
    optimalValues[current_bomb] = (int *)malloc(sizeof(int)*size*size);
    initMatrix(size, optimalValues[current_bomb], INFINITY);
    for(k = size; k > 0; k--) {
      for(j = 0; j < k; j++) {
        i = j + size - k;
        if(i - j <= current_bomb) {
          optimalValues[current_bomb][j*size+i] = 0;
        }
        else {
          for(bomb_pos = j+1; bomb_pos < i+1; bomb_pos++) {
            optimalValues[current_bomb][j*size+i] = getMinimum(
              optimalValues[current_bomb][j*size+i],
              optimalValues[0][j*size+(bomb_pos-1)]+optimalValues[current_bomb-1][bomb_pos*size+i]
            );
          }
        }
      }
    }
    getOptimalValues(size, values, current_bomb+1, total_bombs, optimalValues);
  } else {
    printf("minimal remaining strategic value: %d\n", optimalValues[total_bombs][size-1]);
  }
}

void lawrence(Dataset* inputs, int count) {
  int i,k;
  for(i = 0; i < count; i++) {
    int size = inputs[i].depots;
    int *optimalValues[inputs[i].attacks];
    optimalValues[0] = (int *)malloc(sizeof(int)*size*size);
    initMatrix(size, optimalValues[0], 0);
    printf("n = %d\nm = %d\n", inputs[i].depots, inputs[i].attacks);
    getStrategicValues(inputs[i].values, size, optimalValues[0]);
    getOptimalValues(size, inputs[i].values, 1, inputs[i].attacks, optimalValues);
    printf("=============================================\n\n");
  }
}
