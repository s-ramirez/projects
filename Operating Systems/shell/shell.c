#include <sys/wait.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>
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

int bg_exec;
char *fileIn, *fileOut;
struct Process *process_input(char *line);
int execute_command(struct Process *proc);

// Builtin CD function
int change_dir(char **args) {
  if (args[1] == NULL) {
    fprintf(stderr, "shell: expected argument\n");
  } else {
    if (chdir(args[1]) != 0) {
      perror("shell");
    }
  }
  return 1;
}

// History functions
char history[1000][MAX_INPUT_SIZE];
int history_count = 0;

void read_history(FILE *fp) {
	rewind(fp);
	history_count = 0;
	while(fgets(history[history_count], MAX_INPUT_SIZE, fp)) {
    history_count++;
	}
}

int print_history() {
	execlp("cat", "cat", "-n", ".myhistory", NULL);
	return 2;
}

char* get_history(int position) {
	position--;
	if(position > history_count || position < 0) {
		printf("Index out of range\n");
		fflush(stdout);
		exit(1);
	} else {
		return history[position];
	}
}

char* check_history(char* line) {
	int pos, found_pos, i;
	char substr[MAX_INPUT_SIZE];
	char *cmd, *tmp;

	// Check for history commands
	for(pos = 0; line[pos] != '\0'; pos++) {
		if(line[pos] == '!') {
			tmp = (char*) malloc(sizeof(char) * MAX_INPUT_SIZE);
			pos++;
			found_pos = pos;
			while(line[pos] != ' ' && line[pos] != '\0' && line[pos] != '\n') {
				substr[pos-found_pos] = line[pos];
				pos++;
			}
			cmd = get_history(atoi(substr));
			// Copy first half
			found_pos--;
			for(i = 0; i < found_pos; i++)
				tmp[i] = line[i];
			// Copy from history
			for(i = 0; i < strlen(cmd)-1; i++)
				tmp[found_pos+i] = cmd[i];
			// Copy second half
			for(i = found_pos+i; pos < strlen(line); pos++ && i++)
				tmp[i] = line[pos];
			tmp[pos] = '\0';
			return tmp;
		}
	}
	line[pos] = '\0';
	return line;
}

int run(char **args) {
	int status;
	if(strcmp(args[0], "cd") == 0) {
		// If the user wants a change of directory
		return change_dir(args);
	} else if (strcmp(args[0], "history") == 0){
		return print_history();
	} else {
		if(fileIn != NULL) {
			int fin = open(fileIn, O_RDONLY);
			dup2(fin, fileno(stdin));
			close(fin);
		}
		status = execvp(args[0], args);
		if(status == -1) {
			printf("%s: Command not found\n", args[0]);
			fflush(stdout);
			return status;
		}
		return status;
	}
}

// Start a new process
int run_process(struct Process *proc) {
	int f_des[2], count;
	pid_t pid;

	if(proc->pipe == NULL) {
		return run(proc->args);
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
				return run(proc->args);
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
			if(fileOut != NULL) {
				int fout = open(fileOut, O_WRONLY|O_CREAT, 0600);
				dup2(fout, fileno(stdout));
				close(fout);
			}
			return run_process(proc);
		break;
		default:
			// Parent process
			if(!bg_exec) {
				do {
					waitpid(pid, &status, WUNTRACED);
				} while (!WIFEXITED(status) && !WIFSIGNALED(status));
			}
		break;
	}
	return 1;
}

// Concat two strings
char* concat(const char *s1, const char *s2)
{
  const size_t len1 = strlen(s1);
  const size_t len2 = strlen(s2);
  char *result = malloc(len1+len2+2);//+2 for the zero-terminator and empty space
  memcpy(result, s1, len1);
	memcpy(result+len1, " ", 1);
  memcpy(result+len1+1, s2, len2+1);//+1 to copy the null-terminator
  return result;
}

// Process user input
struct Process *process_input(char *line) {
	char *token;
	int position, str_started;
	struct Process *newProc;
	struct Process *proc = (struct Process*) malloc(sizeof(struct Process));
	proc->argc = 0;
	proc->args = (char**) malloc(MAX_INPUT_SIZE*sizeof(char));

	bg_exec = 0;
	fileOut = NULL;
	fileIn = NULL;
	str_started = 0;

	token = strtok(line, " ");
	while(token != NULL) {
		switch(token[0]) {
			case '|':
				// Pipe
				proc->argc++;
				proc->args[position] = '\0';

				newProc = (struct Process*) malloc(sizeof(struct Process));
				newProc->pipe = proc;
				newProc->argc = 0;
				newProc->args = (char**) malloc(MAX_INPUT_SIZE*sizeof(char));

				proc = newProc;
				position = 0;
			break;
			case '>':
				// Redirect output to file
				proc->argc++;
				proc->args[position] = '\0';

				fileOut = strtok(NULL, " ");
			break;
			case '<':
				// Redirect input to file
				proc->argc++;
				proc->args[position] = '\0';

				fileIn = strtok(NULL, " ");
			break;
			case '&':
				// Background execution
				bg_exec = 1;
			break;
			case '\'':
			case '\"':
				// Quotation marks
				str_started = 1;
			default:
				proc->argc++;
				proc->args[position] = token;
				if (str_started){
					token = strtok(NULL, " ");
					while(token != NULL){
						proc->args[position] = concat(proc->args[position], token);
						if(token[strlen(token)-1] == '\"' || token[strlen(token)-1] == '\'')
							break;
						token = strtok(NULL, " ");
					}
					// Remove first and last characters
					proc->args[position]++;
					proc->args[position][strlen(proc->args[position])-1] = '\0';
				}
				position++;
			break;
		}
		token = strtok(NULL, " ");
	}
	proc->args[position] = '\0';
	return proc;
}

// Read the users input
int read_input(char* line) {
	char buffer;

	int position = 0;
	int history_nav = history_count;

	while(read(0, &buffer, 1) >= 0) {
		if(buffer > 0) {
			// If a escape character was read
			switch(buffer) {
			 case ESCAPE:
				read(0, &buffer, 1);
				read(0, &buffer, 1);

				if(buffer == 'A') { // Up arrow
					if(history_nav -1 >= 0) {
						history_nav--;

						while(position > 0) {
							write(2, "\10\33[1P", 5);
							position--;
						}
						int i;
						for(i = 0; history[history_nav][i] != '\n'; i++) {
							position = i;
							line[i] = history[history_nav][i];
							write(2, &line[i], 1);
						}
						position++;
					}
        } else if(buffer == 'B') { // Down arrow
					while(position > 0) {
						write(2, "\10\33[1P", 5);
						position--;
					}
					if(history_nav + 1 < history_count) {
						history_nav++;

						int i;
						for(i = 0; history[history_nav][i] != '\n'; i++) {
							position = i;
							line[i] = history[history_nav][i];
							write(2, &line[i], 1);
						}
						position++;
					} else if(history_nav + 1 == history_count) {
						history_nav = history_count;
					}
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
	int status = 1;
	char *line = (char*) malloc(sizeof(char)*MAX_INPUT_SIZE);
	struct Process *commands, *temp;
	FILE *history_file;

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
	history_file = fopen(".myhistory", "a+");
	read_history(history_file);

	do {
		if(USER != NULL){
			printf(USER_COLOR "%s" COLOR_RESET ":" PATH_COLOR "%s" COLOR_RESET " $ ", USER, HOST);
		} else {
			printf("$ ");
		}
		fflush(stdout);
		// Read the user's input
		read_input(line);
		fflush(stdout);

		if(strlen(line) != 0){
			// Check for commands from history
			line = check_history(line);

			fprintf(history_file, "%s\n", line);
			fflush(history_file);
			read_history(history_file);
			// Parse input
			commands = process_input(line);

			// Execute commands
			status = execute_command(commands);

			while(commands != NULL) {
				temp = commands;
				commands = commands->pipe;

				free(temp->args);
				free(temp);
			}
		}
	} while(status);
	// Return the terminal's state to the original
	fclose(history_file);
	tcsetattr(0,TCSANOW, &termios_save);
}

int main(int argc, char **argv) {
	input_loop();
  return 0;
}
