#include "common.h"

#define TIMEOUT 20

#define ME 0 /* My node id */
#define N 1 /* Number of nodes */
#define REQ_NUM 2 /* Nodes sequence number */
#define HIGHEST_REQ_NUM 3 /* Highest request number */
#define OUTSTANDING_REPLY 4 /* Number of outstanding replies */
#define REQUEST_CS 5 /* True when node requests CS */
#define NODES 100 /* Other nodes */
#define REPLY_DEFFERED 200 /* Reply_deferred[i] is true when node defers reply to node i */

// Concat two strings
char* concat(const char *s1, const char *s2)
{
  char *result = malloc(strlen(s1)+strlen(s2)+1);//+1 for the zero-terminator
  //in real code you would check for errors in malloc here
  strcpy(result, s1);
  strcat(result, s2);
  return result;
}

// Open requested queue
int open_queue(int id){
  key_t key = ftok(".", id);
  return msgget(key, IPC_CREAT | 0660);
}

int *sharedmem, mutex_sem, wait_sem, nodes_sem, replyq, requestq, printerq;

int get_node(int id) {
  int j;
  for(j = 1; j < sharedmem[N]; j++) {
    if(sharedmem[NODES+j] == id)
      return j;
  }
  return -1;
}

/* Send a message */
int send_msg(int to, int queue, int type, char content[MAX_SIZE]) {
  struct msgbuf msg;

  msg.type = type;
  msg.msg_to = to;
  msg.msg_fm = sharedmem[ME];
  msg.req_num = sharedmem[REQ_NUM];
  strcpy(msg.content, content);

  return msgsnd(queue, &msg, MSG_SIZE, 0);
}

/* k is the sequence number being requested */
/* i is the node making the request */
void request_handler(int k, int i) {
  int defer_it, node_index;

  printf("[*] Node %d wants to write\n", i);
  if(k > sharedmem[HIGHEST_REQ_NUM])
    sharedmem[HIGHEST_REQ_NUM] = k;

  P(mutex_sem);
    node_index = get_node(i);
    defer_it = sharedmem[REQUEST_CS] &&
            ((k > sharedmem[REQ_NUM]) ||
              (k == sharedmem[REQ_NUM] && i > sharedmem[ME]));
  V(mutex_sem);
  /* Defer_it is true if we have priority */
  if(defer_it) {
    printf("\t> I have priority, he should wait.\n");
    sharedmem[REPLY_DEFFERED + node_index] = 1;
  } else {
    //send reply message
    printf("\t> He has priority, answering back...\n");
    send_msg(i, replyq, REPLY, "");
  }
}

void printer_handler() {
  int i;
  printf("[*] I want to write\n");
  P(mutex_sem);
    sharedmem[REQUEST_CS] = 1;
    sharedmem[REQ_NUM] = sharedmem[HIGHEST_REQ_NUM]++;
  V(mutex_sem);
  sharedmem[OUTSTANDING_REPLY] = sharedmem[N] - 1;
  // Send request to all
  printf("\t> Requesting everyone for permission...\n");
  for(i=1; i < sharedmem[N]; i++) {
    printf("\t> Requesting %d for permission.\n", sharedmem[NODES + i]);
    send_msg(sharedmem[NODES + i], requestq, REQUEST, "");
  }
  // Wait for replies
  printf("\t> Waiting for permission... \n");
  while(sharedmem[OUTSTANDING_REPLY] != 0)
    P(wait_sem);
  printf("\t> Permission obtained! Writing... \n");
  // CRITICAL SECTION
  char buffer[MAX_SIZE];
  snprintf(buffer, sizeof(buffer), "### START OUTPUT FOR NODE %i ###", sharedmem[ME]);
  send_msg(PRINTER_QUEUE, printerq, REQUEST, buffer);
  sleep(5);
  memset(buffer,0,sizeof(buffer));
  snprintf(buffer, sizeof(buffer), "--- END OUTPUT FOR NODE %i ---", sharedmem[ME]);
  send_msg(PRINTER_QUEUE, printerq, REQUEST, buffer);
  // END CRITICAL SECTION

  sharedmem[REQUEST_CS] = 0;
  printf("\t> Writing done! Notifying everyone... \n");
  int node;
  for (i = 1; i < sharedmem[N]; i++) {
    if (sharedmem[REPLY_DEFFERED + i]) {
      sharedmem[REPLY_DEFFERED + i] = 0;
      send_msg(sharedmem[NODES+i], replyq, REPLY, "");
    }
  }
  printf("\t> Done writing! \n");
}

void reply_handler() {
  sharedmem[OUTSTANDING_REPLY] = sharedmem[OUTSTANDING_REPLY]-1;
  V(wait_sem);
}

int main(int argc, char *argv[]) {
  int memid, status, timer;
  pid_t pid = getpid();

  // Initialize EZIPC
  SETUP();

  if(argc < 2) {
    printf("One argument expected: <node_number>\n");
    exit(1);
  }

  int nodenum = atoi(argv[1]);

  /* Get shared memory segment */
  memid = shmget(pid, MAX_SIZE, 0640|IPC_CREAT);
  CHECK(memid != -1);

  /* Open queues */
  replyq = open_queue(REPLY_QUEUE);
  CHECK(replyq != -1);
  requestq = open_queue(REQUEST_QUEUE);
  CHECK(requestq != -1);
  printerq = open_queue(PRINTER_QUEUE);
  CHECK(printerq != -1);

  /* Create semaphores */
  mutex_sem = SEMAPHORE(SEM_BIN, 1);
  wait_sem = SEMAPHORE(SEM_BIN, 1);
  nodes_sem = SEMAPHORE(SEM_BIN, 1);

  /* Attach the segment */
  sharedmem = shmat(memid, (void *) 0, 0);
  CHECK(sharedmem != (int *) -1);

  sharedmem[ME] = nodenum;

  /* Send initial message */
  struct msgbuf ack_msg, mrecv;

  status = send_msg(ACK, requestq, ACK, "");
  CHECK(status != -1);

  /* Join the group */
  printf("[*] Sending initial message...\n");

  sharedmem[NODES] = nodenum;
  if(nodenum == 1) {
    printf("\t> I\'m the first node \n");
    sharedmem[N] = 1;
    sharedmem[HIGHEST_REQ_NUM] = 0;
  } else {
    status = msgrcv(replyq, &ack_msg, MSG_SIZE, sharedmem[ME], 0);
    CHECK(status != -1);
    printf("\t> Found a sponsor, with id: %i who sent %s\n", ack_msg.msg_fm, ack_msg.content);
    char* token;
    int i;
    token = strtok(ack_msg.content, " ");
    sscanf(token, "%d", &i);
    sharedmem[N] = i + 1;
    token = strtok(NULL, " ");
    sscanf(token, "%d", &i);
    sharedmem[HIGHEST_REQ_NUM] = i;

    i = 1;
    token = strtok(NULL, " ");

    while(token != NULL) {
      sscanf(token, "%d", &sharedmem[NODES + i]);
      token = strtok(NULL, " ");
      i++;
    }
    /* Let everyone know I'm here */
    for (i = 1; i < sharedmem[N]; i++) {
      send_msg(sharedmem[NODES+i], requestq, ACK, argv[1]);
    }
  }

  /* Start communication */
  int request_proc, reply_proc, broadcast_proc;
  request_proc = fork();
  CHECK(request_proc != -1);

  if(request_proc == 0) {
    // In child first child process
    // Monitor request queue
    printf("[*] Request process ready\n");
    while(1) {
      status = msgrcv(requestq, &mrecv, MSG_SIZE, sharedmem[ME], 0);
      CHECK(status != -1);
      printf("[*] Received request from node %d...\n", mrecv.msg_fm);
      switch(mrecv.type) {
        case ACK:
          printf("[*] Acknowledged node: %s\n", mrecv.content);
          P(nodes_sem);
            sharedmem[NODES + sharedmem[N]] = atoi(mrecv.content);
            sharedmem[N]++;
          V(nodes_sem);
        break;
        default:
          request_handler(mrecv.req_num, mrecv.msg_fm);
        break;
      }
    }
  } else {
    // In parent
    // Spawn second child to monitor reply queue
    reply_proc = fork();
    CHECK(reply_proc != -1);

    if(reply_proc == 0) {
      //Monitor reply queue
      printf("[*] Reply process ready\n");
      while(1) {
        status = msgrcv(replyq, &mrecv, MSG_SIZE, sharedmem[ME], 0);
        CHECK(status != -1);

        printf("[*] Received reply from node %d\n", mrecv.msg_fm);
        reply_handler();
      }
    } else {
      //Spawn a third child to monitor broadcast requests
      broadcast_proc = fork();
      CHECK(broadcast_proc != -1);
      int pos, cur;
      char buffer[MAX_SIZE];

      if(broadcast_proc){
        //Monitor request queue for broadcast messages
        printf("[*] Broadcast process ready\n");
        while(1) {
          status = msgrcv(requestq, &mrecv, MSG_SIZE, ACK, 0);
          CHECK(status != -1);
          if(mrecv.msg_fm != sharedmem[ME]){
            P(nodes_sem);
              printf("\t> A new node with id: %d has entered, sponsoring...\n", mrecv.msg_fm);
              buffer[0] = sharedmem[N] + '0';
              buffer[1] = ' ';
              buffer[2] = sharedmem[HIGHEST_REQ_NUM] + '0';
              pos = 3;
              for(cur = 0; cur < sharedmem[N]; cur++) {
                buffer[pos] = ' ';
                buffer[pos+1] = sharedmem[NODES+cur] + '0';
                pos += 2;
              }
              buffer[pos] = '\0';
              printf("\t> Sending %s to new node %d\n", buffer, mrecv.msg_fm);
            V(nodes_sem);
            send_msg(mrecv.msg_fm, replyq, ACK, buffer);
          }
        }
      } else {
        //Parent
        //Write every once in a while
        while (1) {
          //timer = GET_SLEEP();
          timer = 20;
          printf("[*] Waiting %ds until trying to write...\n", timer);
          sleep(timer);
          P(nodes_sem);
            printer_handler();
          V(nodes_sem);
        }
      }
    }
  }

  /* Remove shared memory */
  shmctl(memid, IPC_RMID, (struct shmid_ds *) 0);
  return 0;
}
