#ifndef COMMON_H_
#define COMMON_H_

/* Includes */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/types.h>
#include <sys/msg.h>

#define PRINTER_QUEUE  1
#define REQUEST_QUEUE  2
#define REPLY_QUEUE  3
#define SEM_MUTEX "/mutex_"
#define SEM_WAIT "/wait_"

#define MAX_SIZE 1024
#define SHM_SIZE 1024
#define BIG_NUM 2147483647
#define MAX_SLEEP 50

#define GET_SLEEP() ((rand() / (BIG_NUM/MAX_SLEEP)) + 1)

#define CHECK(x) \
  do { \
    if (!(x)) { \
      fprintf(stderr, "%s:%d: ", __func__, __LINE__); \
      perror(#x); \
      exit(-1); \
    } \
  } while (0) \

struct msgbuf {
  long int msg_to; /* type of message */
  long int msg_fm; /* message from */
  char content[MAX_SIZE]; /* text of message */
};

#endif
