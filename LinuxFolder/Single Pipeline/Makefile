# Compiler
CC = g++
# Compiler flags
CFLAGS = -std=c++11 -Wall
# Libraries required by OpenCV
LIBS = `pkg-config --libs opencv4`
# Include path for OpenCV
INC_PATH = -I/usr/local/include/opencv4

# Default target
all: Single_Pipeline_Test

# Target for building the executable
Single_Pipeline_Test: main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# Target for compiling main.cpp into an object file
main.o: main.cpp
	$(CC) $(CFLAGS) $(INC_PATH) $(LIBS) -c -o $@ $<

# Target for cleaning up object files and the executable
clean:
	rm -f *.o Single_Pipeline_Test
