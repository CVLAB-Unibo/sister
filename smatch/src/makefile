CC=g++
OPENMP= -fopenmp
CFLAGS=-I. -msse4.1 -msse4.2 -O3 -ffast-math  -march=core2 $(DEST_ARCH) $(OPENMP) -Wno-write-strings
LDFLAGS= $(DEST_ARCH) $(OPENMP) `pkg-config opencv --cflags --libs`
ODIR=../obj
DEPS = types.h stereoalgo.h
_OBJ = run.o census.o sgm.o postprocess.o DSI.o 
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

run: $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ run
