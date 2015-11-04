#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 256

#define N 9999     // number of bodies
#define MASS 0     // row in array for mass
#define X_POS 1    // row in array for x position
#define Y_POS 2    // row in array for y position
#define Z_POS 3    // row in array for z position
#define X_VEL 4    // row in array for x velocity
#define Y_VEL 5    // row in array for y velocity
#define Z_VEL 6    // row in array for z velocity
#define G 10       // "gravitational constant" (not really)
#define MU 0.001   // "frictional coefficient"
#define BOXL 100.0 // periodic boundary box length

__global__ void initRandomStates(unsigned int seed, curandState_t* states);
__global__ void calculateTimestep(float* dev_body, float *dev_nextbody, curandState_t* states);
__global__ void initPosAndVel(float *dev_body, curandState_t* states);

int main(int argc, char **argv) {
  int tmax = 0;

  curandState_t* states;
  float *dev_body, *dev_nextbody;

  if (argc != 2) {
    fprintf(stderr, "Format: %s { number of timesteps }\n", argv[0]);
    exit (-1);
  }
  tmax = atoi(argv[1]);
  cudaSetDevice(1);

  int total_size = sizeof(float) * N * 7;
  //Data array of results, stores each set of results in positions according to the timesteps
  float *body[tmax];

  //Allocate memory for the random states in GPU
  cudaMalloc((void**) &states, sizeof(curandState_t) * N);

  //Calculate the number of required blocks
  int reqBlocks = (int) ceil((float) N/BLOCK_SIZE);

  //Initialize all of the random states on the GPU
  initRandomStates<<<reqBlocks, BLOCK_SIZE>>>(time(NULL), states);
  cudaThreadSynchronize();

  //Initialize bodies with random positions and velocities
  cudaMalloc((void**) &dev_body, total_size);
  initPosAndVel<<<reqBlocks, BLOCK_SIZE>>>(dev_body, states);
  cudaThreadSynchronize();

  //Allocate host memory for the initial values
  body[0] = (float*) malloc(total_size);
  //Copy the initial values into the results matrix, 0 indicates the initial timestep
  cudaMemcpy(body[0], dev_body, total_size, cudaMemcpyDeviceToHost);
  //Allocate the table "nextbody" used when calculating time steps
  cudaMalloc((void**) &dev_nextbody, total_size);
  //Cycle through each time step
  for (int t = 0; t < tmax; t++) {
    //In order to use only two matrices for the process, they are swapped every other time step
    if(t%2 == 0){
      //Calculate the next time step using values from dev_body, storing results in dev_nextbody
      calculateTimestep<<<reqBlocks, BLOCK_SIZE>>>(dev_body, dev_nextbody, states);
      cudaThreadSynchronize();

      //Allocate space for a new timestep in the results
      body[t+1] = (float*) malloc(total_size);
      //Copy the recently obtained results into the newly allocated space
      cudaMemcpy(body[t+1], dev_nextbody, total_size, cudaMemcpyDeviceToHost);
    } else {
      //Calculate the next time step using values from dev_nextbody, storing results in dev_body
      calculateTimestep<<<reqBlocks, BLOCK_SIZE>>>(dev_nextbody, dev_body, states);
      cudaThreadSynchronize();

      //Allocate space for a new timestep in the results
      body[t+1] = (float *) malloc(total_size);
      //Copy the recently obtained results into the newly allocated space
      cudaMemcpy(body[t+1], dev_body, total_size, cudaMemcpyDeviceToHost);
    }
  }

  //Print out positions in PDB format, going through each timestep
  for(int t = 0; t < tmax; t++) {
    printf("MODEL %8d\n", t+1);
    for (int i = 0; i < N; i++) {
      int index = i*7;
  	  printf("%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n",
        "ATOM", i+1, "CA ", "GLY", "A", i+1, body[t][index + X_POS], body[t][index + Y_POS], body[t][index + Z_POS], 1.00, 0.00);
    }
    printf("TER\nENDMDL\n");
  }
  //because we care
  cudaFree(dev_body);
  cudaFree(dev_nextbody);

  //Graceful
  exit(0);
}

__global__ void initRandomStates(unsigned int seed, curandState_t* states) {
  //Obtain the global id of the thread
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int globalId = row*blockDim.x + col;

  //Initialize the currand using the desired seed
  curand_init(seed, globalId, 0, &states[globalId]);
}

__global__ void initPosAndVel(float *dev_body, curandState_t* states) {
  //Obtain the global id of a thread in order to know if it should be accessed
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int globalId = row*blockDim.x + col;

  if(globalId < N) {
    int objectIndex = globalId * 7;
    //Access each object (each on is made up by 7 floats)
    //Initialize the object's with a constant mass
    dev_body[objectIndex + MASS] = 0.001;

    //Initialize positions and velocities using randomly generated numbers
    dev_body[objectIndex + X_POS] = curand_uniform(&states[globalId]);
    dev_body[objectIndex + Y_POS] = curand_uniform(&states[globalId]);
    dev_body[objectIndex + Z_POS] = curand_uniform(&states[globalId]);

    dev_body[objectIndex + X_VEL] = curand_uniform(&states[globalId]);
    dev_body[objectIndex + Y_VEL] = curand_uniform(&states[globalId]);
    dev_body[objectIndex + Z_VEL] = curand_uniform(&states[globalId]);
  }
}

__global__ void calculateTimestep(float* dev_body, float *dev_nextbody, curandState_t* states) {
  float dt = 0.05; // time interval
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int globalId = row*blockDim.x + col;

  if(globalId < N){
    //Get the current objects index by multiplying its global id by the size of each object (7 floats)
    int objectIndex = globalId * 7;
    float Fx_dir, Fy_dir, Fz_dir;
    Fx_dir = 0.0;
    Fy_dir = 0.0;
    Fz_dir = 0.0;
    float x_diff, y_diff, z_diff;

    for (int x = 0; x < N; x++) {  //Calculate force on body due to element in position x
      if (globalId != x) {
        //calculate position difference between body i and x in x-,y-, and z-directions
        int xIndex = x * 7;

        x_diff = dev_body[xIndex + X_POS] - dev_body[objectIndex + X_POS];
        y_diff = dev_body[xIndex + Y_POS] - dev_body[objectIndex + Y_POS];
        z_diff = dev_body[xIndex + Z_POS] - dev_body[objectIndex + Z_POS];

        // periodic boundary conditions
        if (x_diff <  -BOXL * 0.5) x_diff += BOXL;
        if (x_diff >=  BOXL * 0.5) x_diff -= BOXL;
        if (y_diff <  -BOXL * 0.5) y_diff += BOXL;
        if (y_diff >=  BOXL * 0.5) y_diff -= BOXL;
        if (z_diff <  -BOXL * 0.5) z_diff += BOXL;
        if (z_diff >=  BOXL * 0.5) z_diff -= BOXL;

        // calculate distance (r)
        float rr = (x_diff * x_diff + y_diff * y_diff + z_diff * z_diff);
        float r = sqrt(rr);

        // force between bodies i and x
        float F = 0;

        // if sufficiently far away, gravitation force
        if (r > 2.0) {
          //Compute gravitational force between body and x
          F = (float) (G*dev_body[objectIndex + MASS]*dev_body[xIndex + MASS])/rr;

          //Compute and add the frictional force
          F += MU*(curand_uniform(&states[globalId])-0.5);

          Fx_dir += F * x_diff / r;  // resolve forces in x and y directions
          Fy_dir += F * y_diff / r;  // and accumulate forces
          Fz_dir += F * z_diff / r;  //
        } else {
          // if too close, weak anti-gravitational force
          float F = G * 0.01 * 0.01 / r;
          Fx_dir -= F * x_diff / r;  // resolve forces in x and y directions
          Fy_dir -= F * y_diff / r;  // and accumulate forces
          Fz_dir -= F * z_diff / r;  //
        }
      }
    }

    // Update postions and velocity in the new array
    dev_nextbody[objectIndex + MASS] = dev_body[objectIndex + MASS];
    //Update velocities
    dev_nextbody[objectIndex + X_VEL] = dev_body[objectIndex + X_VEL] + (float)(Fx_dir*dt)/dev_body[objectIndex + MASS];
    dev_nextbody[objectIndex + Y_VEL] = dev_body[objectIndex + Y_VEL] + (float)(Fy_dir*dt)/dev_body[objectIndex + MASS];
    dev_nextbody[objectIndex + Z_VEL] = dev_body[objectIndex + Z_VEL] + (float)(Fz_dir*dt)/dev_body[objectIndex + MASS];

    if (dev_nextbody[objectIndex + X_VEL] <  -BOXL * 0.5) dev_nextbody[objectIndex + X_VEL] += BOXL;
    if (dev_nextbody[objectIndex + X_VEL] >=  BOXL * 0.5) dev_nextbody[objectIndex + X_VEL] -= BOXL;
    if (dev_nextbody[objectIndex + Y_VEL] <  -BOXL * 0.5) dev_nextbody[objectIndex + Y_VEL] += BOXL;
    if (dev_nextbody[objectIndex + Y_VEL] >=  BOXL * 0.5) dev_nextbody[objectIndex + Y_VEL] -= BOXL;
    if (dev_nextbody[objectIndex + Z_VEL] <  -BOXL * 0.5) dev_nextbody[objectIndex + Z_VEL] += BOXL;
    if (dev_nextbody[objectIndex + Z_VEL] >=  BOXL * 0.5) dev_nextbody[objectIndex + Z_VEL] -= BOXL;

    //Update positions
    dev_nextbody[objectIndex + X_POS] = dev_body[objectIndex + X_POS] + dev_nextbody[objectIndex + X_VEL]*dt;
    dev_nextbody[objectIndex + Y_POS] = dev_body[objectIndex + Y_POS] + dev_nextbody[objectIndex + Y_VEL]*dt;
    dev_nextbody[objectIndex + Z_POS] = dev_body[objectIndex + Z_POS] + dev_nextbody[objectIndex + Z_VEL]*dt;

    // periodic boundary conditions
    if (dev_nextbody[objectIndex + X_POS] <  -BOXL * 0.5) dev_nextbody[objectIndex + X_POS] += BOXL;
    if (dev_nextbody[objectIndex + X_POS] >=  BOXL * 0.5) dev_nextbody[objectIndex + X_POS] -= BOXL;
    if (dev_nextbody[objectIndex + Y_POS] <  -BOXL * 0.5) dev_nextbody[objectIndex + Y_POS] += BOXL;
    if (dev_nextbody[objectIndex + Y_POS] >=  BOXL * 0.5) dev_nextbody[objectIndex + Y_POS] -= BOXL;
    if (dev_nextbody[objectIndex + Z_POS] <  -BOXL * 0.5) dev_nextbody[objectIndex + Z_POS] += BOXL;
    if (dev_nextbody[objectIndex + Z_POS] >=  BOXL * 0.5) dev_nextbody[objectIndex + Z_POS] -= BOXL;
  }
}
