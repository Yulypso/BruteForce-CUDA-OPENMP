CC_FLAGS = -Wall
LIB_FLAGS = -Xpreprocessor -fopenmp
BIN = BIN

FILE = breaker_for.c
NB_THREADS = 8
IS_CANCELLATION = true

all: compile run clean

compile: 
	clang $(CC_FLAGS) -o $(BIN) $(LIB_FLAGS) $(FILE) -lomp

clean:
	rm -rf $(BIN)

run:
	OMP_NUM_THREADS=$(NB_THREADS) OMP_CANCELLATION=$(IS_CANCELLATION) ${ENV} ./$(BIN)