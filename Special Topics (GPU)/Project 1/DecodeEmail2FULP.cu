#include <stdio.h>

__global__ void decode(char *message, char *result);

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
      char *file_content, *result_content;
      long file_size;

      //Get size of the file, then return to begining in order to read it after
      fseek(file, 0, SEEK_END);
      file_size = ftell(file);
      fseek(file, 0, SEEK_SET);

      //Read the file to memory
      file_content = (char *) malloc(file_size);
      result_content = (char *) malloc(file_size);

      int position = 0;
      char c;
      while ((c = fgetc(file)) != EOF)
      {
          file_content[position++] = (char)c;
      }
      file_content[position] = '\0';

      //Decode the message
      char *dev_encoded;
      char *dev_decoded;

      cudaMalloc((void **)&dev_decoded, file_size);
      cudaMalloc((void **)&dev_encoded, file_size);

      cudaMemcpy(dev_encoded, file_content, file_size, cudaMemcpyHostToDevice);

      int length = strlen(file_content);

      decode<<<length, 1>>>(dev_encoded, dev_decoded);

      cudaThreadSynchronize();
      cudaMemcpy(result_content, dev_decoded, file_size, cudaMemcpyDeviceToHost);

      cudaFree(dev_encoded);
      cudaFree(dev_decoded);

      printf("Message:\n%s\n", result_content);
      exit (0);
    }
  }
  exit(0);
}

__global__ void decode(char *message, char *result) {
  int position = blockIdx.x;
  result[position] = message[position] - 1;
}
