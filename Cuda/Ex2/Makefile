BIN = BIN
FILE = err1.cu
# FILE = err2.cu

all: compile run clean

compile: 
	@nvcc $(FILE) -o $(BIN) 

run:
	./$(BIN)

clean:
	@rm -rf $(BIN)