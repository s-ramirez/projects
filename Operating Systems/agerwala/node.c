#include "common.h"
//
int main(int argc, char **argv) {}
// #include <semaphore.h>
// #include <sys/ipc.h>
// #include <sys/shm.h>
//
// #include "common.h"
//
// #define N 0 /* Number of nodes */
// #define REQ_NUM 1 /* Nodes sequence number */
// #define HIGHEST_REQ_NUM 2 /* Highest request number */
// #define OUTSTANDING_REPLY 3 /* Number of outstanding replies */
// #define REQUEST_CS 4 /* True when node requests CS */
// #define REPLY_DEFFERED 5 /* Reply_deferred[i] is true when node defers reply to node i */
//
// int nodenum;
//
// char* concat(const char *s1, const char *s2)
// {
//     char *result = malloc(strlen(s1)+strlen(s2)+1);//+1 for the zero-terminator
//     //in real code you would check for errors in malloc here
//     strcpy(result, s1);
//     strcat(result, s2);
//     return result;
// }
//
// int main(int argc, char *argv[]) {
//   int memid, *sharedmem, status;
//   int key = getuid();
//
//   if(argc < 2) {
//     printf("One argument expected: <node_number>\n");
//     exit(1);
//   }
//
//   nodenum = atoi(argv[1]);
//   /* Get shared memory segment */
//   memid = shmget(key+nodenum, SHM_SIZE, 0640|IPC_CREAT);
//   CHECK(memid != -1);
//
//   /* Create semaphores */
//
//   sem_t *sem_mutex = sem_open(concat(SEM_MUTEX,argv[1]), O_CREAT, 0644, 0);
//   CHECK(sem_mutex > 0);
//   sem_t *sem_wait = sem_open(concat(SEM_WAIT,argv[1]), O_CREAT, 0644, 0);
//   CHECK(sem_wait > 0);
//   /* Attach the segment */
//   sharedmem = shmat(memid, (void *) 0, 0);
//   CHECK(sharedmem != -1);
//
//   /* Write shared variables */
//
//   /* Detach shared memory */
//   //status = shmdt(sharedmem);
//   //CHECK(status != -1);
//   return 0;
// }
//
// /* k is the sequence number being requested */
// /* i is the node making the request */
// int request_handler() {
//   int defer_it;
//
//
//
// }
//
// int reply_handler() {
//
// }
