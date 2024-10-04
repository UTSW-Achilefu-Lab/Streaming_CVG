/*
 * Receiver.cpp: Receiver for OpenCV_GStreamer example
 *
 * Copyright (C) 2019 Simon D. Levy
 *
 * MIT License
 */

#include <opencv2/opencv.hpp>
using namespace cv;

#include <iostream>
using namespace std;

int main()
{
    printf("\n\n--------------------------------- PROGRAM STARTED --------------------------------- \r\n");

    // The sink caps for the 'rtpjpegdepay' need to match the src caps of the 'rtpjpegpay' of the sender pipeline
    // Added 'videoconvert' at the end to convert the images into proper format for appsink, without
    // 'videoconvert' the receiver will not read the frames, even though 'videoconvert' is not present
    // in the original working pipeline

    // Original
    // string recieveFromPiPipeline = "udpsrc port=5001 ! application/x-rtp, encoding-name=H264,payload=96 ! rtph264depay ! decodebin ! videoconvert ! appsink sync=false drop=true";

    // Testing
    string recieveFromPiPipeline = "udpsrc port=5001 ! application/x-rtp ! rtph264depay ! decodebin ! videoconvert ! appsink sync=false drop=true";

    // Ip address for RPI needed here
    std::string sendBackToPiPipeline = "appsrc ! videoconvert ! queue ! vaapih264enc ! h264parse !  rtph264pay ! udpsink host=172.17.141.124 port=5003";

    VideoCapture cap(recieveFromPiPipeline, CAP_GSTREAMER);
    cv::VideoWriter RPIWriter(sendBackToPiPipeline, cv::VideoWriter::fourcc('H', '2', '6', '4'), 30, cv::Size(3280, 2464), true);

    if (!cap.isOpened())
    {
        cerr << "VideoCapture not opened" << endl;
        exit(-1);
    }

    while (true)
    {

        Mat frame;

        cap.read(frame);

        imshow("Frames Recieved From RPI", frame);

        if (waitKey(1) == 27)
        {
            break;
        }
    }

    return 0;
}