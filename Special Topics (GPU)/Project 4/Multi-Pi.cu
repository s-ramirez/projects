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
void destroyStreams(int device, cudaStream_t* streams, char* buffer, int** dev_count, char** dev_read);
int executeCount(int device, cudaStream_t* streams, int file_size, int file_read, char* buffer, int** dev_count, char** dev_read, FILE* file, int** results);

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
    int i;

    //Declare and allocate results arrays, one for each device
    int **results0 = (int**) malloc(NUM_STREAMS*sizeof(int*));
    int **results1 = (int**) malloc(NUM_STREAMS*sizeof(int*));

    //Declare and allocate stream arrays, one for each device
    cudaStream_t* streams0 = (cudaStream_t*) malloc(NUM_STREAMS*sizeof(cudaStream_t));
    cudaStream_t* streams1 = (cudaStream_t*) malloc(NUM_STREAMS*sizeof(cudaStream_t));

    //Declare and allocate a set of buckets for counts and a reading buffer, one for each device
    int **dev_count0 = (int **)malloc(NUM_STREAMS*sizeof(int*));
    char **dev_read0 = (char **)malloc(NUM_STREAMS*sizeof(char*));

    int **dev_count1 = (int **)malloc(NUM_STREAMS*sizeof(int*));
    char **dev_read1 = (char **)malloc(NUM_STREAMS*sizeof(char*));

    //Execute the code once so that everything is initialized before meassuring
    initializeStreams(0, streams0, dev_count0, dev_read0);
    initializeStreams(1, streams1, dev_count1, dev_read1);
    
    //Synchronize all streams
    //This shouldn't count in the total sync count since it's done in order to get a more precise meassure
    syncStreams(0, streams0);
    syncStreams(1, streams1);

    //Start the timer
    start = clock();

    //Obtain the file length
    fseek(file, 0, SEEK_END);
		file_size = ftell(file);
		fseek(file, 0, SEEK_SET);

    //Allocate a page locked buffer for reading the file for each device
    char* buffer0, *buffer1;
    cudaSetDevice(0);
    cudaHostAlloc((void **)&buffer0, file_size*sizeof(char), cudaHostAllocDefault);
    cudaSetDevice(1);
    cudaHostAlloc((void **)&buffer1, file_size*sizeof(char), cudaHostAllocDefault);

    //Initialize results buckets and the file reading buffer for first device
    initializeStreams(0, streams0, dev_count0, dev_read0);
    initializeStreams(1, streams1, dev_count1, dev_read1);
    
    //Initialize host results arrays
    for(i = 0; i < NUM_STREAMS; i++) {
      results0[i] = (int*) malloc(sizeof(int)*10);    
      results1[i] = (int*) malloc(sizeof(int)*10);    
    }    

    //Set the initial device
    int currentDevice = 0;

    while(file_read < file_size) { //While read characters are less than the total file size
      currentDevice = currentDevice % 2; //Reinitialize the current device if greater than two (number of gpus)
      //Execute a counting cicle using the set device and its associated values
      if(currentDevice == 0) {
        file_read = executeCount(currentDevice, streams0, file_size, file_read, buffer0,
          dev_count0, dev_read0, file, results0);
      } else {
        file_read = executeCount(currentDevice, streams1, file_size, file_read, buffer1,
          dev_count1, dev_read1, file, results1);
      }
      currentDevice++;
		}
    //After reading the whole file, synchronize all streams
    syncStreams(0, streams0);
    syncStreams(1, streams1);
    //Increase and print the number of synchronizations
    num_syncs++;
    printf("GPUs Synchronized %d\n", num_syncs);
    //Close the opened file
		fclose(file);
    //Save results to file
    saveToFile(results0, results1);
    //Because we care, free all the used cuda variables and destroy the streams
    cudaSetDevice(0);
    destroyStreams(0, streams0, buffer0, dev_count0, dev_read0);
    cudaSetDevice(1);
    destroyStreams(1, streams1, buffer1, dev_count1, dev_read1);
    //Stop the timer and print the total elapsed time
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
  //Create each stream and its associated reading buffer and count buckets
  for(i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreate(&streams[i]);
    cudaMalloc((void**)&dev_read[i], N*sizeof(char));
    cudaMalloc((void**)&dev_count[i], 10*sizeof(int));
    //Start each bucket in 0
    initCounts<<<10, 1, 0, streams[i]>>>(dev_count[i]);
  }
}

void syncStreams(int device, cudaStream_t* streams) {
  //Synchronize the desired device
  cudaSetDevice(device);
  cudaDeviceSynchronize();
}

void destroyStreams(int device, cudaStream_t* streams, char* buffer, int** dev_count, char** dev_read) {
  int i;
  cudaSetDevice(device);
  //Free all the variables associated to each stream in the desired device and destroy the stream
  for(i = 0; i < NUM_STREAMS; i++) {
    cudaFree(dev_read[i]);
    cudaFree(dev_count[i]);
    cudaStreamDestroy(streams[i]);
  }
  cudaFree(buffer);
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

  for(i = 0; i < NUM_STREAMS; i++) { //Do the following for each stream in the set device
    if((file_size - (file_read + N)) < 0) { //If there are not N unread characters in the file
      size[i] = file_size - file_read; //Read only the remaining characters
    } else {
      size[i] = N;//Read N characters
    }
    fread(&buffer[file_read], size[i], 1, file);//Read a chunk of 'size' into the device's buffer at the current position
    //Begin copying asynchronously this buffer into the current stream
    cudaMemcpyAsync(dev_read[i], buffer+file_read, size[i]*sizeof(char), cudaMemcpyHostToDevice, streams[i]);
    //Increment the amount of read characters by the size determined before
    file_read += size[i];
    if(size[i] != N) { //Don't go further to other streams if there is nothing else to read
      break;
    }
  }

  for(i = 0; i < NUM_STREAMS; i++) {//Do the following for each stream in the set device
    //Count the digits in the current's stream chunk of read characters
    countDigits<<<size[i], 1, 0, streams[i]>>>(dev_read[i], dev_count[i], size[i]);
    if(size[i] != N) { //Don't go further to other streams if there is nothing else to read
      break;
    }
  }

  for(i = 0; i < NUM_STREAMS; i++) {//Do the following for each stream in the set device
    //Asynchronously copy results from the stream back to the host
    cudaMemcpyAsync(results[i], dev_count[i], 10*sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
  }
  return file_read; //Return the current number of read characters
}

__global__ void initCounts(int* dev_buckets) {
  //Initialize each count in 0
  int position = blockIdx.x;
  dev_buckets[position] = 0;
}

__global__ void countDigits(char *dev_read, int* dev_buckets, int size) {
  //Use a derivation of "Bucket sort" to count each digit from the chunk of read bytes
  int position = blockIdx.x;
  if(position < size){
    //Each thread gets one character from the reading buffer
    int value = dev_read[position] - '0'; //Substract '0' in order to obtain the int representation of the character
    if(value >= 0) { //If the obtained representation is below 0 it might be an empty space or line break, then ignore
      atomicAdd(&dev_buckets[value], 1); //Use an atomic operation to increase the character's bucket
    }
  }
}

