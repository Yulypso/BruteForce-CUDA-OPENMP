BIN = BIN
FILE = matrices.cu

all: compile run clean

compile: 
	@nvcc $(FILE) -o $(BIN) 

run:
	./$(BIN)

clean:
	@rm -rf $(BIN)