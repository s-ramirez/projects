# Compiler
CC = gcc
# Compiler flags
CFLAGS = -lrt -pthread

TARGETS = printer hacker node

all: $(TARGETS)

.o:
	$(CC) -o $@ $(CFLAGS)

clean:
	$(RM) $(TARGETS)
