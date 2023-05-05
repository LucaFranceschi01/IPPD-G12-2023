
CC=mpicc

CFLAGS= -g -Wall -Icommon

STENCIL_COMMON_SRC=common/stencil/printarr_par.c
STENCIL_CFLAGS= $(CFLAGS) -Icommon/stencil
BINS=stencil stencil_alltoallw

all: $(BINS)

stencil: stencil.c $(STENCIL_COMMON_SRC)
	$(CC) $(STENCIL_CFLAGS) $^ -o $@ -lm

stencil_alltoallw: stencil_alltoallw.c $(STENCIL_COMMON_SRC)
	$(CC) $(STENCIL_CFLAGS) $^ -o $@ -lm

clean:
	rm -f $(BINS)
	rm -f output*.bmp
