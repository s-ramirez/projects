#include "common.h"


int main(int argc, char **argv)
{
  int status, timer;
  mqd_t printerq;
  char buffer[MAX_SIZE] = {"Hacker: Pura vida?"};

  /* open the mail queue */
  printerq = mq_open(PRINTER_QUEUE, O_WRONLY);
  CHECK((mqd_t)-1 != printerq);

  printf("Starting \"hacker\" program...\n");

  do {
    timer = GET_SLEEP();
    printf("Waiting %ds until next message...\n", timer);
    sleep(timer);
    printf("I don\'t follow the rules!\n");
    fflush(stdout);

    /* user input */
    //memset(buffer, 0, MAX_SIZE);
    //fgets(buffer, MAX_SIZE, stdin);

    /* send the message */
    status  = mq_send(printerq, buffer, MAX_SIZE, 0);
    CHECK(status >= 0);
    printf("Message sent!\n");
  } while (1);

  /* cleanup */
  CHECK((mqd_t)-1 != mq_close(printerq));

  return 0;
}
