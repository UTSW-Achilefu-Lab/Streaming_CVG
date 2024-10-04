# Compiler
CC = g++
# Compiler flags
CFLAGS = -std=c++11 -Wall
# Libraries required by OpenCV
LIBS = `pkg-config --libs opencv4`
# Include path for OpenCV
INC_PATH = -I/usr/local/include/opencv4

# Default target
all: Test_Dual_GStreamer_Pipelines

# Target for building the executable
Test_Dual_GStreamer_Pipelines: main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# Target for compiling main.cpp into an object file
main.o: main.cpp
	$(CC) $(CFLAGS) $(INC_PATH) -c -o $@ $<

# Target for cleaning up object files and the executable
clean:
	rm -f *.o Test_Dual_GStreamer_Pipelines
