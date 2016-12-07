#include <iostream>
#include <cstdio>
#include <unistd.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>

int main(int argc, char *argv[]) {
  int key = getuid();
  int memid;
  if(argc != 2) {
    printf("One argument expected: <node ID>")
  }
  if ((memid = shmget(key+argv[1], 1000, 0640|IPC_CREAT)) == -1) {
    perror("shmget memid");
    return 1;
  }
  return 0;
}
