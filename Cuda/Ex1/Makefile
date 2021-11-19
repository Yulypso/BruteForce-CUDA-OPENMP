BIN = BIN
FILE = Ex1.cu

all: compile run clean

compile: 
	@nvcc $(FILE) -o $(BIN) 

run:
	./$(BIN)

clean:
	@rm -rf $(BIN)