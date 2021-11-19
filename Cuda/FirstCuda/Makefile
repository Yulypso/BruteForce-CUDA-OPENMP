BIN = BIN
FILE = firstCuda.cu

all: compile run clean

compile: 
	@nvcc $(FILE) -o $(BIN) 

run:
	./$(BIN)

clean:
	@rm -rf $(BIN)