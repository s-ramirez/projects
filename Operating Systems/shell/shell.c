#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ncurses.h>

#define USER_COLOR	"\x1b[1;32m"
#define PATH_COLOR "\x1b[1;34m"
#define COLOR_RESET   "\x1b[0m"
#define MAX_INPUT_SIZE 380

// Read the users input
void readUserInput() {
	const char* PATH = getenv("PATH");
	const char* USER = getenv("USER");

	char cwd[1024];
	char input[MAX_INPUT_SIZE];
	char identifiers[MAX_INPUT_SIZE];

	while(1){
		getcwd(cwd, sizeof(cwd));
		if(USER != NULL && cwd != NULL){
			printf(USER_COLOR "%s" COLOR_RESET ":" PATH_COLOR "%s" COLOR_RESET " $ ", USER, cwd);
		} else {
			printf("$ ");
		}
	}
}

int main() {
	//readUserInput();
	int ch;
	if (ch = getc(stdin)) {
   /* This is a special key. Get the second byte */
   printf("The second byte is %d\n", ch);
 }
	return 0;
}
