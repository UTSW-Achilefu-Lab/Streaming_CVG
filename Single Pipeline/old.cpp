#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
using namespace std;
using namespace cv;

/*
    Terminal commands to use
        - To send back to the raspberry pi : raspivid -t 0 -n -w 640 -h 480 -fps 25 -b 2000000 -o - | gst-launch-1.0 -v fdsrc ! h264parse ! rtph264pay config-interval=10 pt=96 ! udpsink host=172.17.141.174 port=5000

        - To receive from the rpi: gst-launch-1.0 -v udpsrc address=172.17.141.124 port=5001 caps=application/x-rtp ! rtph264depay ! h264parse ! queue ! v4l2h264dec ! autovideosink sync=false

        - To specify which camera to use : raspivid -t 0 -n -w 640 -h 480 -fps 25 -b 2000000 -o - -cs <camera_number> | gst-launch-1.0 -v fdsrc ! h264parse ! rtph264pay config-interval=10 pt=96 ! udpsink host=172.17.141.174 port=5000


*/

int main()
{
    printf("\n\n--------------------------------- PROGRAM STARTED --------------------------------- \r\n");
    // Set up GStreamer pipeline for receiving video stream from Raspberry Pi
    std::string pipeline = "udpsrc port=5000 ! application/x-rtp, encoding-name=H264,payload=96 ! rtph264depay ! decodebin ! videoconvert ! appsink sync=false drop=true";

    std::string sendToPiPipeline = "appsrc ! videoconvert ! queue ! vaapih264enc ! h264parse !  rtph264pay ! udpsink host=172.17.141.124 port=5001";

    /*
        Decoder testing
            - libav:  avenc_h264_omx: libav OpenMAX IL H.264 video encoder encoder [NOPE]
                - [ WARN:0@6.415] global cap_gstreamer.cpp:2730 writeFrame OpenCV | GStreamer warning: Error pushing buffer to GStreamer pipeline
            - openh264:  openh264enc: OpenH264 video encoder
                - Original one used with raspicam and worked fine
                - Delay still present
            - vaapi:  vaapih264enc: VA-API H264 encoder [NOPE]
                - Significant delay 

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
    VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    cv::VideoWriter RPIWriter(sendToPiPipeline, cv::VideoWriter::fourcc('H', '2', '6', '4'), 25, cv::Size(640, 480), true);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open video stream" << std::endl;
        return -1;
    }
    else
        cout << "Video capture open " << endl;
    // Main loop to read frames from the stream
    while (true)
    {
        Mat frame;
        cap >> frame;

        if (frame.empty())
        {
            std::cerr << "Received empty frame" << std::endl;
            break;
        }

        // Perform your image processing on 'frame' here using OpenCV functions

        // Display the processed frame
        // imshow("Processed Frame", frame);

        // Send the processed frame to the RPI
        RPIWriter.write(frame);
        imshow("Captured Frame From RPI", frame);
        // Check for key press to exit
        if (waitKey(1) == 27)
        {
            break;
        }
    }

    // Release the VideoCapture object and close any OpenCV windows
    cap.release();
    destroyAllWindows();

    return 0;
}