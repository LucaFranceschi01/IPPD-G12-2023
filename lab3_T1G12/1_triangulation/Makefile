CC=pgcc
CFLAGS=-fast -Minfo=accel
OBJ=delaunay


gpu:
	$(CC) main.c $(OBJ).c -ta=tesla:managed -o $(OBJ) $(CFLAGS)

cpu:
	$(CC) main.c $(OBJ).c -ta=multicore -o $(OBJ) $(CFLAGS)

clean:
	rm $(OBJ)
