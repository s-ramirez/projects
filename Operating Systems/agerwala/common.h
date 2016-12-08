#ifndef COMMON_H_
#define COMMON_H_

/* Includes */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <semaphore.h>

#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/types.h>
#include <sys/msg.h>
#include <sys/shm.h>

#include "ezipc.h"

#define PRINTER_QUEUE  1
#define REQUEST_QUEUE  2
#define REPLY_QUEUE  3

#define MAX_SIZE 1024
#define MSG_SIZE 1024 + sizeof(long int)
#define BIG_NUM 2147483647
#define MAX_SLEEP 50

/* Message types*/
#define ACK 100
#define REQUEST 101
#define REPLY 102
#define CONF 103

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
  int type; /* message type */
  int req_num; /* request number */
  char content[MAX_SIZE]; /* text of message */
};

#endif
