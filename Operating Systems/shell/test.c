#include <sys/wait.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>

#define READ 0
#define WRITE 1

int test() {
  int pipe1[2];
  pipe(pipe1);
  // Child process
  if(fork() > 0) {
      // Parent
      dup2(pipe1[READ], READ);
      close(pipe1[READ]);
      close(pipe1[WRITE]);
      return execlp("grep", "grep", "ps", NULL);
  } else {
    // Child
    dup2(pipe1[WRITE], WRITE);
    close(pipe1[WRITE]);
    close(pipe1[READ]);
    return execlp("ps", "ps", "-A", NULL);
  }
}

int main(int argc, char* argv[])
{
    pid_t pid;
    int pipe1[2], status;

  	pid = fork();
  	if(pid < 0)
  			perror("Fork");
  	else if(pid == 0){
			test();
    }
    else {
      do {
          waitpid(pid, &status, WUNTRACED);
        } while (!WIFEXITED(status) && !WIFSIGNALED(status));
    }
  	return 1;

    // pipe(fd);
    // pid = fork();
    //
    // if(pid==0)
    // {
    //     printf("i'm the child used for ls \n");
    //     dup2(fd[WRITE], STDOUT_FILENO);
    //     close(fd[WRITE]);
    //     close(fd[READ]);
    //     execlp("ps", "ps", "-A", NULL);
    // }
    // else
    // {
    //     pid=fork();
    //
    //     if(pid==0)
    //     {
    //         printf("i'm in the second child, which will be used to run grep\n");
    //         dup2(fd[READ], STDIN_FILENO);
    //         close(fd[READ]);
    //         close(fd[WRITE]);
    //         execlp("grep", "grep", "ps", NULL);
    //     }
    // }

    return 0;
}
