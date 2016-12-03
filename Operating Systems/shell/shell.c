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
int run_process(char **args) {
	pid_t pid, wpid;
	int status;

	pid = fork();
	if(pid == 0) {
		// Child process
		if (execvp(args[0], args) == -1) {
			perror("Exec");
		}
		exit(EXIT_FAILURE);
	} else if(pid < 0) {
		perror("Fork");
	} else {
		do {
      wpid = waitpid(pid, &status, WUNTRACED);
    } while (!WIFEXITED(status) && !WIFSIGNALED(status));
	}
	return 1;
}

// Execute command
int execute_command(struct Process *proc) {
	pid_t pid;
	int f_des[2];

	if()
	if (pipe(f_des) == -1) {
		perror("Pipe");
		return 1;
	}
	switch(fork()){
		case -1:
			perror("Pipe");
		return 1;
		case 0: //Child
			dup2(f_des[WRITE], fileno(stdout))
	}
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
	struct Process *proc = malloc(sizeof(struct Process));
	proc->args = malloc(sizeof(line));

	token = strtok(line, " ");
	while(token != NULL) {
		switch(token[0]) {
			case '|':
				newProc = malloc(sizeof(struct Process));
				newProc->pipe = proc;
				newProc->args = malloc(sizeof(line));

				proc = newProc;
				position = 0;
			break;
			case '>':
				// redirect to file
			break;
			default:
				proc->args[position] = token;
				position++;
			break;
		}
		token = strtok(NULL, " ");
	}
	return proc;
}

// Read the users input
char *read_input() {
	char *line = (char*) malloc(sizeof(char)*MAX_INPUT_SIZE);
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
					return line;
				break;
				default:
					line[position] = buffer;
					position++;
	        write(2,&buffer,1);
				break;
			}
		}
	}
	return line;
}

void input_loop() {
	int status;
	char *line;
	struct Process *commands;

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

	do {
		if(USER != NULL){
			printf(USER_COLOR "%s" COLOR_RESET ":" PATH_COLOR "%s" COLOR_RESET " $ ", USER, HOST);
		} else {
			printf("$ ");
		}
		fflush(stdout);
		// Read the user's input
		line = read_input();
		printf("line: %s\n", line);
		fflush(stdout);
		// Parse input
		commands = process_input(line);
		while(commands != NULL) {
			printf("Args: %s %s\n", commands->args[0], commands->args[1]);
			fflush(stdout);
			commands = commands->pipe;
		}

		// Execute commands
		// status = execute_command(commands);

		free(line);
		free(commands);
	} while(status);
	// Return the terminal's state to the original
	tcsetattr(0,TCSANOW, &termios_save);
}

int main(int argc, char **argv)
{
	input_loop();
  return 0;
}
