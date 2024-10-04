cmake_minimum_required(VERSION 3.10)
project(MockCVGProgram)

set(CMAKE_CXX_STANDARD 14)

# Add the path to WiringPi library
set(WIRINGPI_PATH /usr/local/lib)

# Find WiringPi library
find_library(WIRINGPI_LIB wiringPi PATHS ${WIRINGPI_PATH})
find_package(OpenCV REQUIRED)

# Check if WiringPi library is found
if(NOT WIRINGPI_LIB)
    message(FATAL_ERROR "WiringPi library not found.")
endif()

# Set the source file
set(SOURCE_FILES main.cpp)

# Create the executable
add_executable(MockCVGProgram ${SOURCE_FILES} yen_threshold.cpp)

target_include_directories(MockCVGProgram PRIVATE 
    ${OpenCV_INCLUDE_DIRS} 
    /home/pi/Desktop/Onboard_VS_Streaming_RPI/Mock CVG Program/
    /home/pi/Desktop/Onboard_VS_Streaming_RPI/Mock CVG Program/include
    )

# Link the WiringPi library
target_link_libraries(MockCVGProgram ${WIRINGPI_LIB} ${OpenCV_LIBS})
