#ifndef COMMON_H_
#define COMMON_H_

/* Includes */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <mqueue.h>

#define PRINTER_QUEUE  "/printer_queue"
#define MAX_SIZE 1024
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


#endif /* #ifndef COMMON_H_ */
