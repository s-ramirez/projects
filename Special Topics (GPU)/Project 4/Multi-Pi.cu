#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 65535
#define NUM_STREAMS 4

//Kernel declarations
__global__ void initCounts(int* buckets);
__global__ void countDigits(char *dev_read, int* dev_buckets, int size);

//Function declarations
void saveToFile(int** results0, int** results1);
void syncStreams(int device, cudaStream_t* streams);
void initializeStreams(int device, cudaStream_t* streams, int** dev_count, char** dev_read);
int executeCount(int device, cudaStream_t* streams, int file_size, int file_read, char* buffer, int **dev_count, char **dev_read, FILE *file, int **results);

int main(int argc, char **argv) {

  if (argc != 2) {
    fprintf(stderr, "Format: %s { name of the file containing digits }\n", argv[0]);
    exit (-1);
  }
  FILE *file = fopen(argv[1], "r");

  if (file == NULL) {
    printf("Error: Could not read the file\n");
    exit(-1);
  }
  else {
      clock_t start, stop;
			int file_size = 0;
			int file_read = 0;
      int num_syncs = 0;

      int **results0 = (int**) malloc(NUM_STREAMS*sizeof(int*));
      int **results1 = (int**) malloc(NUM_STREAMS*sizeof(int*));

      cudaStream_t* streams0 = (cudaStream_t*) malloc(NUM_STREAMS*sizeof(cudaStream_t));
      cudaStream_t* streams1 = (cudaStream_t*) malloc(NUM_STREAMS*sizeof(cudaStream_t));

      int **dev_count0 = (int **)malloc(NUM_STREAMS*sizeof(int));
      char **dev_read0 = (char **)malloc(NUM_STREAMS*sizeof(char));

      int **dev_count1 = (int **)malloc(NUM_STREAMS*sizeof(int));
      char **dev_read1 = (char **)malloc(NUM_STREAMS*sizeof(char));

      //Execute the code once so that everything is initialized before meassuring
      initializeStreams(0, streams0, dev_count0, dev_read0);
      initializeStreams(1, streams1, dev_count1, dev_read1);

      //Start the timer
      start = clock();

      //Obtain the file length
      fseek(file, 0, SEEK_END);
			file_size = ftell(file);
			fseek(file, 0, SEEK_SET);

      //Allocate a page locked buffer for reading the file
      char* buffer;
      cudaHostAlloc((void **)&buffer, file_size*sizeof(char), cudaHostAllocDefault);

      //Initialize results buckets and the file reading buffer for first device
      initializeStreams(0, streams0, dev_count0, dev_read0);
      initializeStreams(1, streams1, dev_count1, dev_read1);

      int currentDevice = 0;
      while(file_read < file_size) {
        currentDevice = currentDevice % 2;
        if(currentDevice == 0) {
          file_read = executeCount(currentDevice, streams0, file_size, file_read, buffer,
            dev_count0, dev_read0, file, results0);
        } else {
          file_read = executeCount(currentDevice, streams1, file_size, file_read, buffer,
            dev_count1, dev_read1, file, results1);
        }
        currentDevice++;
			}
      syncStreams(0, streams0);
      syncStreams(1, streams1);
      num_syncs++;
      printf("GPUs Synchronized %d\n", num_syncs);
			fclose(file);
      //Save results to file
      saveToFile(results0, results1);

      stop = clock();
      float elapsed_time = (float)(stop - start)/CLOCKS_PER_SEC;
      printf("Elapsed Time: %2.8fs\n", elapsed_time);
  }
  //Graceful
  exit(0);
}

void initializeStreams(int device, cudaStream_t* streams, int** dev_count, char** dev_read) {
  int i;
  cudaSetDevice(device);
  for(i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreate(&streams[i]);
    cudaMalloc((void**)&dev_read[i], N*sizeof(char));
    cudaMalloc((void**)&dev_count[i], 10*sizeof(int));
    initCounts<<<10, 1, 0, streams[i]>>>(dev_count[i]);
  }
  syncStreams(device, streams);
}

void syncStreams(int device, cudaStream_t* streams) {
  int i;
  cudaSetDevice(device);
  for(i = 0; i < NUM_STREAMS; i++) {
    cudaStreamSynchronize(streams[i]);
  }
}

void saveToFile(int** results0, int** results1) {
  //Save to frequencies to file
  int k, i, sum = 0;
  int final_results[10];
  FILE *file;
  //Open the freq.dat file
  file = fopen("freq.dat","w");
  if(file == NULL){
    //There was an error opening the file
    printf("Error: The file freq.dat could not be opened\n");
  } else {
    //Add all results into a single "final_results" array
    for(k = 0; k < 10; k++) {
      int value = 0;
      for(i = 0; i < NUM_STREAMS; i++) {
        value += results0[i][k];
        value += results1[i][k];
      }
      final_results[k] = value;
      //Add to a total sum of all values, used for normalizing
      sum += value;
    }
    //Print the normalized results to the freq.dat file
    for(k = 0; k < 10; k++) {
      fprintf(file, "%d\t%2.8f\n", k, (float)final_results[k]/sum);
    }
  }
}

int executeCount(int device, cudaStream_t* streams, int file_size, int file_read, char* buffer,
  int **dev_count, char **dev_read, FILE *file, int **results) {
  cudaSetDevice(device);
  int size[NUM_STREAMS], i;

  for(i = 0; i < NUM_STREAMS; i++) {
    if((file_size - (file_read + N)) < 0) {
      size[i] = (int)file_size - file_read;
    } else {
      size[i] = N;
    }
    fread(&buffer[file_read], size[i], 1, file);
    cudaMemcpyAsync(dev_read[i], buffer+file_read, size[i]*sizeof(char), cudaMemcpyHostToDevice, streams[i]);
    file_read += size[i];
  }

  for(i = 0; i < NUM_STREAMS; i++) {
    countDigits<<<size[i], 1, 0, streams[i]>>>(dev_read[i], dev_count[i], size[i]);
  }

  for(i = 0; i < NUM_STREAMS; i++) {
    results[i] = (int*) malloc(sizeof(int)*10);
    cudaMemcpyAsync(results[i], dev_count[i], 10*sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
  }
  return file_read;
}

__global__ void initCounts(int* dev_buckets) {
  int position = blockIdx.x;
  dev_buckets[position] = 0;
}

__global__ void countDigits(char *dev_read, int* dev_buckets, int size) {
  int position = blockIdx.x;
  if(position < size){
    int value = dev_read[position] - '0';
    if(value >= 0) {
      atomicAdd(&dev_buckets[value], 1);
    }
  }
}

