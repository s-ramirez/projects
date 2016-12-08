#include "common.h"

int create_queue(int id) {
  key_t key = ftok(".", id);
  return msgget(key, IPC_CREAT | 0660);
}

int main(int argc, char **argv)
{
    struct msgbuf mrecv;
    int printerq, replyq, requestq, status;
    char buffer[MAX_SIZE + 1];

    /* create the message queue */

    printerq = create_queue(PRINTER_QUEUE);
    CHECK(printerq != -1);
    requestq = create_queue(REQUEST_QUEUE);
    CHECK(requestq != -1);
    replyq = create_queue(REPLY_QUEUE);
    CHECK(replyq != -1);

    printf("Starting printer service...\n");
    while(1){

      /* receive the message */
      status = msgrcv(printerq, &mrecv, MSG_SIZE, PRINTER_QUEUE, 0);
      CHECK(status != -1);

      printf("%s\n", mrecv.content);
    }

    /* Remove message queues */
    msgctl(printerq, IPC_RMID, (struct msqid_ds *) 0);
    msgctl(requestq, IPC_RMID, (struct msqid_ds *) 0);
    msgctl(replyq, IPC_RMID, (struct msqid_ds *) 0);

    return 0;
}
