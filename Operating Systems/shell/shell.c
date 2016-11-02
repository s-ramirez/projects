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

#define ESCAPE '\33'
#define BACKSPACE '\177'
#define DELETE '~'
#define ENTER '\n'

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
			perror("shell");
		}
		exit(EXIT_FAILURE);
	} else if(pid < 0) {
		perror("shell");
	} else {
		do {
      wpid = waitpid(pid, &status, WUNTRACED);
    } while (!WIFEXITED(status) && !WIFSIGNALED(status));
	}
	return 1;
}

// Execute command
int execute_command(char **args) {
	if(args[0] == NULL) {
		// Ignore an empty command
		return 1;
	}

	if(strcmp(args[0], "cd") == 0) {
		// If the user wants a change of directory
		return change_dir(args);
	}

	return run_process(args);
}

// Process user input
char **process_input(char *line){
	int position = 0;
	char **tokens = malloc(MAX_INPUT_SIZE * sizeof(char*));
	char *token;

	// VALIDAR malloc
	token = strtok(line, " \t");
	while(token != NULL) {
		tokens[position] = token;
		position++;

		token = strtok(NULL, " \t");
	}
	tokens[position] = NULL;
	return tokens;
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
}

void input_loop() {
	int status;
	char *line;
	char ** tokens;

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
		if(USER != NULL && HOST != NULL){
			printf(USER_COLOR "%s" COLOR_RESET ":" PATH_COLOR "%s" COLOR_RESET " $ ", USER, HOST);
		} else {
			printf("$ ");
		}
		fflush(stdout);
		// Read the user's input
		line = read_input();
		printf("\n");
		fflush(stdout);
		// Parse input
		tokens = process_input(line);
		// Execute commands
		status = execute_command(tokens);

		free(line);
		free(tokens);
	} while(status);
	// Return the terminal's state to the original
	tcsetattr(0,TCSANOW, &termios_save);
}

int main(int argc, char **argv)
{
	input_loop();
  return 0;
}
