#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  if(argc != 2) {
    printf("Error: You must provide the length of the columns/rows\n");
  }
  else {
    int size = atoi(argv[1]);
    if(size >= 0) {

    } else {
        printf("Error: The size provided must be a positive number\n");
    }
  }
  //Graceful
  exit(0);
}