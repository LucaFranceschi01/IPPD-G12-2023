CC=mpicc
CFLAGS= -g -Wall

BINS=bug1 bug2 bug3 bug4 bug5

all: $(BINS)

bug1: bug1.c
	$(CC) $^ -o $@ -lm

bug2: bug2.c
	$(CC) $^ -o $@ -lm

bug3: bug3.c
	$(CC) $^ -o $@ -lm

bug4: bug4.c
	$(CC) $^ -o $@ -lm

bug5: bug5.c
	$(CC) $^ -o $@ -lm

clean:
	rm -f $(BINS)
