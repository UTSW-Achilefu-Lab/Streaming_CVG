# Compiler
CC = g++
# Compiler flags
CFLAGS = -std=c++11 -Wall
# Libraries required by OpenCV
LIBS = `pkg-config --libs opencv4`
# Include path for OpenCV and additional headers
INC_PATH = -I/usr/local/include/opencv4 -I"/home/utsw-bmen-laptop/Coding Folder/Onboard_VS_Streaming_LINUX/Two Pipelines Multithread/"

# Default target
all: GStreamerTest_LINUX_LAPTOP_IMAGE_PROCESSED

# Target for building the executable
GStreamerTest_LINUX_LAPTOP_IMAGE_PROCESSED: main.o yen_thresholding.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# Target for compiling main.cpp into an object file
main.o: main.cpp yen_threshold.h
	$(CC) $(CFLAGS) $(INC_PATH) -c -o $@ $<

# Target for compiling yen_thresholding.cpp into an object file
yen_thresholding.o: yen_thresholding.cpp yen_threshold.h
	$(CC) $(CFLAGS) $(INC_PATH) -c -o $@ $<

# Target for cleaning up object files and the executable
clean:
	rm -f *.o GStreamerTest_LINUX_LAPTOP_IMAGE_PROCESSED
