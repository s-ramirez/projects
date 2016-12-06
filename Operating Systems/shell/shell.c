#include <sys/wait.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <termios.h>
#include <string.h>

#define USER_COLOR	"\x1b[1;32m"
#define PATH_COLOR "\x1b[1;34m"
#define COLOR_RESET   "\x1b[0m"
#define MAX_INPUT_SIZE 256

#define READ 0
#define WRITE 1

#define ESCAPE '\33'
#define BACKSPACE '\177'
#define DELETE '~'
#define ENTER '\n'

struct Process {
	int argc;
	char **args;
	struct Process *pipe;
};

// Builtin CD function
int change_dir(char **args)
{
  if (args[1] == NULL) {
    fprintf(stderr, "shell: expected argument\n");
  } else {
    if (chdir(args[1]) != 0) {
      perror("shell");
    }
  }
  return 1;
}

// Start a new process
int run_process(struct Process *proc) {
	int f_des[2], count;
	pid_t pid;

	if(proc->pipe == NULL) {
		return execvp(proc->args[0], proc->args);
	} else {
		pipe(f_des);
		switch(fork()) {
			case -1:
				perror("Fork");
				return -1;
			break;
			case 0:
				// Child process
				dup2(f_des[WRITE], WRITE);
				close(f_des[WRITE]);
        close(f_des[READ]);
				return run_process(proc->pipe);
			break;
			default:
				// Parent process
				dup2(f_des[READ], READ);
				close(f_des[READ]);
				close(f_des[WRITE]);
				return execvp(proc->args[0], proc->args);
			break;
		}
	}
	return 1;
}

// Execute command
int execute_command(struct Process *proc) {
	pid_t pid;
	int status;

	pid = fork();
	switch(pid) {
		case -1:
			perror("Fork");
		break;
		case 0:
			// Child process
			return run_process(proc);
		break;
		default:
			// Parent process
			do {
				waitpid(pid, &status, WUNTRACED);
			} while (!WIFEXITED(status) && !WIFSIGNALED(status));
		break;
	}
	return 1;
}

// if(strcmp(args[0], "cd") == 0) {
// 	// If the user wants a change of directory
// 	return change_dir(args);
// }

// Process user input
struct Process *process_input(char *line) {
	int position;
	char *token;
	struct Process *newProc;
	struct Process *proc = (struct Process*) malloc(sizeof(struct Process));
	proc->argc = 0;
	proc->args = (char**) malloc(1*sizeof(char));
	token = strtok(line, " ");
	while(token != NULL) {
		switch(token[0]) {
			case '|':
				proc->argc++;
				proc->args = realloc(proc->args, proc->argc*sizeof(char));
				proc->args[position] = '\0';

				newProc = (struct Process*) malloc(sizeof(struct Process));
				newProc->pipe = proc;
				newProc->argc = 0;
				newProc->args = (char**) malloc(1*sizeof(char));

				proc = newProc;
				position = 0;
			break;
			case '>':
				// redirect to file
			break;
			default:
				proc->argc++;
				proc->args = realloc(proc->args, proc->argc*sizeof(char));
				proc->args[position] = token;
				position++;
			break;
		}
		token = strtok(NULL, " ");
	}
	proc->args = realloc(proc->args, proc->argc*sizeof(char));
	proc->args[position] = '\0';
	return proc;
}

// Read the users input
int read_input(char* line) {
	char buffer;

	int position = 0;

	while(read(0, &buffer, 1) >= 0) {
		if(buffer > 0) {
			// If a escape character was read
			switch(buffer) {
			 case ESCAPE:
				read(0, &buffer, 1);
				read(0, &buffer, 1);

				if(buffer == 'A') {
          write(2, "up", 2); // Up
        } else if(buffer == 'B') {
					write(2, "down", 4); // Down
        } else if(buffer == 'C') {
          write(2, "\33[C", 3); // Right
        } else if(buffer == 'D') {
          write(2, "\10", 2); // Left
        }
				break;
	    	case BACKSPACE:
					if(position > 0) {
	      		write(2, "\10\33[1P", 5);
						position--;
					}
				break;
	    	case DELETE:
	        write(2, "\33[1P", 4);
	    	break;
				case ENTER:
					line[position] = '\0';
					printf("\n");
					fflush(stdout);
					return 1;
				break;
				default:
					line[position] = buffer;
					position++;
	        write(2,&buffer,1);
				break;
			}
		}
	}
	return 0;
}

void input_loop() {
	int status;
	char *line = (char*) malloc(sizeof(char)*MAX_INPUT_SIZE);
	struct Process *commands, *temp;
	FILE *history;

	struct termios termios_p;
	struct termios termios_save;

	// Obtain the current user from environmental variables
	char HOST[1024];
	HOST[1023] = '\0';
	gethostname(HOST, 1023);
	const char* USER = getenv("USER");

	// Store the current terminal's state
	tcgetattr(0, &termios_p);
	termios_save = termios_p;

	// Set echo and canonical mode flags
	termios_p.c_lflag &= ~(ICANON|ECHO);

	// Set terminal to the version with the new flags immediately
	tcsetattr(0,TCSANOW, &termios_p);

	// Open the history file
	history = fopen(".myhistory", "a");

	do {
		if(USER != NULL){
			printf(USER_COLOR "%s" COLOR_RESET ":" PATH_COLOR "%s" COLOR_RESET " $ ", USER, HOST);
		} else {
			printf("$ ");
		}
		fflush(stdout);
		// Read the user's input
		read_input(line);
		fprintf(history, "%s\n", line);
		fflush(history);
		fflush(stdout);

		// Parse input
		commands = process_input(line);

		// Execute commands
		status = execute_command(commands);
		int i;
		while(commands != NULL) {
			temp = commands;
			commands = commands->pipe;

			free(temp->args);
			free(temp);
		}
	} while(status);
	// Return the terminal's state to the original
	fclose(history);
	tcsetattr(0,TCSANOW, &termios_save);
}

int main(int argc, char **argv)
{
	input_loop();
  return 0;
}
