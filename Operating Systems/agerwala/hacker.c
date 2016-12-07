 #include "common.h"

int main(int argc, char **argv)
{
  int printerq;
  pid_t hacker_pid;
  int status, timer;
  struct msgbuf msg;
  char buffer[MAX_SIZE] = {"Hacker: Pura vida?"};

  /* Open the printer queue */
  key_t key = ftok(".", PRINTER_QUEUE);
  printerq = msgget(key, IPC_CREAT | 0660);
  CHECK(printerq != -1);

  printf("Starting \"hacker\" program...\n");

  while (1) {
    msg.msg_to = PRINTER_QUEUE;
    msg.msg_fm = hacker_pid;

    //timer = GET_SLEEP();
    timer = 10;
    printf("Waiting %ds until next message...\n", timer);
    sleep(timer);
    printf("I don\'t follow the rules!\n");

    /* user input */
    //memset(buffer, 0, MAX_SIZE);
    //fgets(buffer, MAX_SIZE, stdin);

    strcpy(msg.content, buffer);

    /* send the message */
    status  = msgsnd(printerq, &msg, MAX_SIZE, 0);
    CHECK(status >= 0);
    printf("Message sent!\n");
  }
  return 0;
}
