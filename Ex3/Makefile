UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
CC_FLAGS = -Wall
LIB_FLAGS = -Xpreprocessor -fopenmp
BIN = BIN

FILE = breaker_for.c
NB_THREADS = 8
IS_CANCELLATION = true

all: compile run clean

compile: 
	@clang $(CC_FLAGS) -o $(BIN) $(LIB_FLAGS) $(FILE) -lomp

clean:
	@rm -rf $(BIN)

run:
	@echo "-- [Running $(FILE) with $(NB_THREADS) threads] --"
	@OMP_NUM_THREADS=$(NB_THREADS) OMP_CANCELLATION=$(IS_CANCELLATION) ./$(BIN)
endif

ifeq ($(UNAME), Linux)
CC_FLAGS = -Wall
LIB_FLAGS = -fopenmp -lcrypt -lm
BIN = BIN

FILE = breaker_for.c
NB_THREADS = 8
IS_CANCELLATION = true

all: compile run clean

compile: 
	@gcc $(CC_FLAGS) $(FILE) -o $(BIN) $(LIB_FLAGS) 

clean:
	@rm -rf $(BIN)

run:
	@echo "-- [Running $(FILE) with $(NB_THREADS) threads] --"
	@OMP_NUM_THREADS=$(NB_THREADS) OMP_CANCELLATION=$(IS_CANCELLATION) ./$(BIN)
endif
