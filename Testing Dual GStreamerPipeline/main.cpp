#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>

#define HORIZONTAL_RESOLUTION 640
#define VERTICAL_RESOLUTION 480
#define FRAME_RATE 30
using namespace std;
using namespace cv;

/*
    Terminal commands to use
        - To send : raspivid -t 0 -n -w 640 -h 480 -fps 25 -b 2000000 -o - | gst-launch-1.0 -v fdsrc ! h264parse ! rtph264pay config-interval=10 pt=96 ! udpsink host=172.17.141.174 port=5000

        - To receive : gst-launch-1.0 -v udpsrc address=172.17.141.124 port=5001 caps=application/x-rtp ! rtph264depay ! h264parse ! queue ! v4l2h264dec ! autovideosink sync=false

        - To specify which camera to use : raspivid -t 0 -n -w 640 -h 480 -fps 25 -b 2000000 -o - -cs <camera_number> | gst-launch-1.0 -v fdsrc ! h264parse ! rtph264pay config-interval=10 pt=96 ! udpsink host=172.17.141.174 port=5000


*/

int main()
{
    std::cout << "Starting Gstreamer Pipeline..." << std::endl;
    // ir camera pipeline

    // Recieve from RPI
    std::string irRecieveCameraPipeline = "udpsrc port=5000 ! application/x-rtp, encoding-name=H264,payload=96 ! rtph264depay ! decodebin ! videoconvert ! appsink sync=false drop=true";
    std::string visibleRecieveFromRPICameraPipeline = "udpsrc port=5001 ! application/x-rtp, encoding-name=H264,payload=96 ! rtph264depay ! decodebin ! videoconvert ! appsink sync=false drop=true";

    // Send back to RPI
    std::string irSendIRCameraPipe = "appsrc ! videoconvert ! queue ! openh264enc bitrate=2000000 complexity=2 ! h264parse !  rtph264pay ! udpsink host=172.17.141.124 port=5002";
    std::string visibleSendIRCameraPipe = "appsrc ! videoconvert ! queue ! openh264enc bitrate=2000000 complexity=2 ! h264parse !  rtph264pay ! udpsink host=172.17.141.124 port=5003";

    /*
        gst-launch-1.0
            - v flag
                - Output status information and property notifications
        appsrc
            - Insert data into GStreamer pipeline
            - Can provide external API functions

        videoconvert
            -
    */

    // Open the video stream
    VideoCapture irCap(irRecieveCameraPipeline, cv::CAP_GSTREAMER);
    VideoCapture visibleCap(visibleRecieveFromRPICameraPipeline, cv::CAP_GSTREAMER);

    // VideoWriter (const String &filename, int apiPreference, int fourcc, double fps, Size frameSize, bool isColor=true)
    cv::VideoWriter irRPIWriter(irSendIRCameraPipe, cv::VideoWriter::fourcc('H', '2', '6', '4'), FRAME_RATE, cv::Size(HORIZONTAL_RESOLUTION, VERTICAL_RESOLUTION), true);
    cv::VideoWriter visibleRPIWriter(visibleSendIRCameraPipe, cv::VideoWriter::fourcc('H', '2', '6', '4'), FRAME_RATE, cv::Size(HORIZONTAL_RESOLUTION, VERTICAL_RESOLUTION), true);

    if (!irCap.isOpened() || !visibleCap.isOpened())
    {
        std::cerr << "Failed to open video stream" << std::endl;
        return -1;
    }
    else
        cout << "Video capture open " << endl;
    while (true)
    {
        Mat irFrame, visibleFrame;
        irCap >> irFrame;
        visibleCap >> visibleFrame;

        if (irFrame.empty() || visibleFrame.empty())
        {
            std::cerr << "Received empty frame" << std::endl;
            break;
        }

        // ------------------------------------------------ Do image processing stuff here ------------------------------------------------ //

        // Display the processed frame
        imshow("ir Frame", irFrame);
        imshow("visible Frame", visibleFrame);

        // Send the processed frame to the RPI
        irRPIWriter.write(irFrame);
        visibleRPIWriter.write(visibleFrame);

        // Check for key press to exit
        if (waitKey(1) == 27)
        {
            break;
        }
    }

    // Release the VideoCapture object and close any OpenCV windows
    irCap.release();
    visibleCap.release();
    destroyAllWindows();

    return 0;
}