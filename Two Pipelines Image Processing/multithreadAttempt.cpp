#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include "yen_threshold.h"
#include <thread>
#include <functional>

#define HORIZONTAL_RESOLUTION 640
#define VERTICAL_RESOLUTION 480
#define FRAME_RATE 30
#define THRESHOLD_WEIGHT 0.1
#define WARPEDFRAME_WEIGHT 0.9
using namespace std;
using namespace cv;

std::mutex irMutex;
std::mutex visibleMutex;

cv::Mat ImgProc_YenThreshold(cv::Mat src, bool compressed, double &foundThresh)
{
    // Convert frame to grayscale
    cv::Mat grey;
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);

    // Histogram constants
    int bins = 256;
    int histSize[] = {bins};
    const float range[] = {0, 256};
    const float *ranges[] = {range};
    int channels[] = {0};

    // equalize
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.7);
    cv::Mat cl;
    clahe->apply(grey, cl);

    // make histogram
    cv::Mat hist;
    cv::calcHist(&cl, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

    // yen_thresholding
    int yen_threshold = Yen(hist);
    foundThresh = yen_threshold;

    // Apply binary thresholding
    cv::Mat thresholded;
    if (compressed)
    {
        cv::threshold(cl, thresholded, double(yen_threshold), 255, cv::THRESH_BINARY);
    }
    else
    {
        cv::threshold(cl, thresholded, double(yen_threshold), 255, cv::THRESH_TOZERO);
    }

    return thresholded;
}

void captureFrames(VideoCapture &cap, Mat *frame, const string &windowName)
{
    while (true)
    {
        cap >> *frame;
        if (frame->empty())
        {
            cerr << "Error: Couldn't read frame from camera" << endl;
            break;
        }
    }
}
int main()
{
    std::cout << "Loading homography" << std::endl;

    // Path to the YAML file
    std::string filename = "/home/utsw-bmen-laptop/Coding Folder/Onboard_VS_Streaming_LINUX/Two Pipelines Multithread/homography_matrix.yaml";
    double foundThresh = 0.0;
    // Open the file using FileStorage
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "Failed to open " << filename << std::endl;
        return -1;
    }

    // Variables to store the camera matrix and distortion coefficients
    // cv::Mat irToVisibleHomography, visibleToIRHomography, zoomedVisibleFrames, zoomedIrFrames, rawCombinedFrames, combinedZoomedFrames, combinedWarpedFrames, visibleToIRProjectedFrame, yenThresholdedFrame;
    cv::Mat irFrames, visibleFrames, combinedFrames, croppedVisibleFrames, croppedIrFrames, irWarpedFrame, visibleWarpedFrame, successfulCalibrationFrameCaptured, warpedFramesSideBySide, zoomedIrFramesGray, visibleFramesGray, visibleToIRProjectedFrame, rawCombinedFrames, visibleFramesBGR, processedFrame, yenThresholdedFrame, combinedZoomedFrames, irToVisibleHomography, visibleToIRHomography;

    // Read the camera matrix and distortion coefficients from the file
    fs["irToVisibleHomography"] >> irToVisibleHomography;
    fs["visibleToIRHomography"] >> visibleToIRHomography;

    // Close the file
    fs.release();

    Mat visibleToIRHomographyMatrix = visibleToIRHomography;
    Mat irToVisibleHomographyMatrix = irToVisibleHomography;

    // Output the read values
    // std::cout
    //     << "\nvisibleToIRHomographyMatrix :\n"
    //     << std::endl
    //     << visibleToIRHomographyMatrix << std::endl;
    // std::cout << "\nirToVisibleHomographyMatrix :\n"
    //           << std::endl
    //           << irToVisibleHomographyMatrix << std::endl;

    std::cout << "Starting Gstreamer Pipeline..." << std::endl;
    // ir camera pipeline

    // Recieve from RPI
    std::string irRecieveCameraPipeline = "udpsrc port=5000 ! application/x-rtp, encoding-name=H264,payload=96 ! rtph264depay ! decodebin ! videoconvert ! appsink sync=false drop=true";
    std::string visibleRecieveFromRPICameraPipeline = "udpsrc port=5001 ! application/x-rtp, encoding-name=H264,payload=96 ! rtph264depay ! decodebin ! videoconvert ! appsink sync=false drop=true";

    // Send back to RPI
    std::string irSendIRCameraPipe = "appsrc ! videoconvert ! openh264enc multi-thread=4 bitrate=2000000 complexity=2 ! h264parse !  rtph264pay ! udpsink host=172.17.141.124 port=5002";
    std::string visibleSendIRCameraPipe = "appsrc ! videoconvert ! openh264enc multi-thread=4 bitrate=2000000 complexity=2 ! h264parse !  rtph264pay ! udpsink host=172.17.141.124 port=5003";

    // Open the video streams
    VideoCapture capIR(irRecieveCameraPipeline, cv::CAP_GSTREAMER);
    VideoCapture visibleCap(visibleRecieveFromRPICameraPipeline, cv::CAP_GSTREAMER);

    // VideoWriter
    cv::VideoWriter irRPIWriter(irSendIRCameraPipe, cv::VideoWriter::fourcc('H', '2', '6', '4'), FRAME_RATE, cv::Size(HORIZONTAL_RESOLUTION, VERTICAL_RESOLUTION), false);
    cv::VideoWriter visibleRPIWriter(visibleSendIRCameraPipe, cv::VideoWriter::fourcc('H', '2', '6', '4'), FRAME_RATE, cv::Size(HORIZONTAL_RESOLUTION, VERTICAL_RESOLUTION), false);

    if (!capIR.isOpened() || !visibleCap.isOpened())
    {
        std::cerr << "Failed to open video stream" << std::endl;
        return -1;
    }
    // Mat visibleFrames, irFrames, irWarpedFrame, visibleWarpedFrame, combineRawImages, combineWarpedImages, visibleFramesBGR;

    Mat *irFramePtr = &irFrames;
    Mat *visibleFramePtr = &visibleFrames;

    // If either irFrames or visibleFrames size isn't initialized here
    irFrames = Mat::zeros(Size(HORIZONTAL_RESOLUTION, VERTICAL_RESOLUTION), CV_8UC3);
    visibleFrames = Mat::zeros(Size(HORIZONTAL_RESOLUTION, VERTICAL_RESOLUTION), CV_8UC3);
    // Create thread

    thread t1(captureFrames, ref(capIR), irFramePtr, "IR Camera");
    thread t2(captureFrames, ref(visibleCap), visibleFramePtr, "Visible Camera");
    Mat visibleGrayFrame, irGrayFrame;
    while (true)
    {

        // imshow("IR Camera Frame On Laptop", *irFramePtr);
        // imshow("Visible Camera Frame On Laptop", *visibleFramePtr);

        cv::cvtColor(*visibleFramePtr, visibleGrayFrame, cv::COLOR_BGR2GRAY);
        cv::cvtColor(*irFramePtr, irGrayFrame, cv::COLOR_BGR2GRAY);

        cv::Rect roiIR(irGrayFrame.cols / 4, irGrayFrame.rows / 4, irGrayFrame.cols / 2, irGrayFrame.rows / 2);
        cv::Rect roiVisible(visibleGrayFrame.cols / 4, visibleGrayFrame.rows / 4, visibleGrayFrame.cols / 2, visibleGrayFrame.rows / 2);

        // Zoom in on both visible and IR frames
        croppedVisibleFrames = visibleFrames(roiVisible);
        croppedIrFrames = irFrames(roiIR);

        if (waitKey(1) == 27) // Escape key
            break;

        visibleRPIWriter.write(croppedVisibleFrames);
        irRPIWriter.write(croppedIrFrames);
    }

    t1.join();
    t2.join();

    // Release the VideoCapture objects and close any OpenCV windows
    capIR.release();
    visibleCap.release();
    destroyAllWindows();

    return 0;
}
