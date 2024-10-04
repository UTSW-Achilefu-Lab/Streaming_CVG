#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include "yen_threshold.h"
#include <thread>
#include <functional>
#include <arpa/inet.h>
#include <unistd.h>
#include <chrono>

#define HORIZONTAL_RESOLUTION 1920
#define VERTICAL_RESOLUTION 1080
#define FRAME_RATE 30
#define THRESHOLD_WEIGHT 0.2
#define WARPEDFRAME_WEIGHT 0.80
using namespace std;
using namespace cv;

std::mutex irMutex;
std::mutex visibleMutex;

void remap_lut_threshold(cv::Mat &src, cv::Mat &dst, float k, int threshold, int &topkvalue)
{
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    cv::Mat hist;
    cv::calcHist(&src, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    int high = 0;
    int low = 0;
    for (int i = 0; i < 256; i++)
    {
        if (hist.at<int>(i) != 0)
        {
            high = i;
            if (low == 0)
                low = i;
        }
    }
    // cout<<"high_low:"<<high<<"\t"<<low<<endl;
    cv::Mat histCumulative = hist.clone();
    for (int i = 1; i < histSize; i++)
    {
        histCumulative.at<float>(i) += histCumulative.at<float>(i - 1);
    }
    float abovePixels = histCumulative.at<float>(255) - histCumulative.at<float>(threshold);
    float totalPixels = src.rows * src.cols;
    topkvalue = 0;
    for (int i = 0; i < histSize; i++)
    {
        if (histCumulative.at<float>(i) >= totalPixels - k * abovePixels)
        {
            topkvalue = i;
            break;
        }
    }
    // cout<<"abovePixels\t"<<abovePixels<<"\ttopkvalue"<<topkvalue<<endl;
    cv::Mat lut(1, 256, CV_8U);
    for (int i = 0; i <= 255; i++)
    {
        if (i > topkvalue)
            lut.at<uchar>(0, i) = 255;
        else if (i > threshold)
            lut.at<uchar>(0, i) = 255.0 * (i - threshold) / (high - threshold);
        else
            lut.at<uchar>(0, i) = 0;
    }
    // cout<<lut<<endl;
    cv::LUT(src, lut, dst);
}

cv::Mat ImgProc_YenThreshold(cv::Mat src, bool compressed, double &foundThresh)
{
    // Not needed for the Streaming application
    // But uncomment if running RPI as 'edge' device
    // Convert frame to grayscale
    // cv::Mat grey;
    // cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);

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
    clahe->apply(src, cl);

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

void captureVisibleFrames(VideoCapture &cap, Mat *frame, const string &windowName)
{
    while (true)
    {
        // auto captureVisibleFrameSTART = std::chrono::high_resolution_clock::now();
        visibleMutex.lock();
        cap >> *frame;
        visibleMutex.unlock();

        if (frame->empty())
        {
            cerr << "Error: Couldn't read frame from camera" << endl;
            break;
        }
        // std::lock_guard<std::mutex> guard(visibleMutex);

        // auto captureVisibleFrameEND = std::chrono::high_resolution_clock::now();
        // auto captureVisibleDURATION = std::chrono::duration_cast<std::chrono::microseconds>(captureVisibleFrameEND - captureVisibleFrameSTART);
        //  std::cout << "Time elapsed for RPI to CAPTURE VISIBLE FRAME : " << captureVisibleDURATION.count() << " microseconds" << std::endl;
    }
    // std::lock_guard<std::mutex> guard(visibleMutex);
}
void captureIRFrames(VideoCapture &cap, Mat *frame, const string &windowName)
{
    cv::Mat flipped;
    while (true)
    {
        irMutex.lock();
        cap >> *frame;
        cv::flip(*frame, *frame, 1);
        irMutex.unlock();

        // cv::flip(*frame, *frame, 1);
        if (frame->empty())
        {
            cerr << "Error: Couldn't read frame from camera" << endl;
            break;
        }
        // std::lock_guard<std::mutex> guard(irMutex);
    }
    // std::lock_guard<std::mutex> guard(irMutex);
}
int main()
{
    std::cout << "Loading homography" << std::endl;

    // Path to the YAML file
    std::string filename = "/home/utsw-bmen-laptop/Onboard_VS_Streaming_LINUX/Two Pipelines Multithread/homography_matrix.yaml";
    double foundThresh = 0.0;
    // Open the file using FileStorage
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "Failed to open " << filename << std::endl;
        return -1;
    }

    // Variables to store the camera matrix and distortion coefficients
    cv::Mat irToVisibleHomography, visibleToIRHomography, zoomedVisibleFrames, zoomedIrFrames, rawCombinedFrames, combinedZoomedFrames, combinedWarpedFrames, visibleToIRProjectedFrame, yenThresholdedFrame, processedFrame, visibleToInfraredHomography;
    cv::Ptr<cv::Mat> ColoredFrame;
    cv::Mat croppedVisibleFrames(HORIZONTAL_RESOLUTION, VERTICAL_RESOLUTION, CV_8UC1); // CV_8UC1 specifies 8-bit unsigned single-channel matrix
    cv::Mat croppedIrFrames(HORIZONTAL_RESOLUTION, HORIZONTAL_RESOLUTION, CV_8UC1);    // CV_8UC1 specifies 8-bit unsigned single-channel matrix
    cv::Mat visibleWarpedFrame(HORIZONTAL_RESOLUTION, HORIZONTAL_RESOLUTION, CV_8UC1); // CV_8UC1 specifies 8-bit unsigned single-channel matrix
    // cv::Mat visibleToInfraredHomography(480, 640, CV_8UC1); // CV_8UC1 specifies 8-bit unsigned single-channel matrix

    Mat visibleFrames, irFrames, irWarpedFrame, combineRawImages, combineWarpedImages, visibleFramesBGR;
    int remapMin = 0;

    // Read the camera matrix and distortion croppedVisibleFrames from the file
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
    // std::string debuggerCameraPipe = "appsrc ! videoconvert ! openh264enc multi-thread=4 bitrate=2000000 complexity=2 ! h264parse !  rtph264pay ! udpsink host=172.17.141.124 port=5004";

    // Open the video streams
    VideoCapture capIR(irRecieveCameraPipeline, cv::CAP_GSTREAMER);
    VideoCapture visibleCap(visibleRecieveFromRPICameraPipeline, cv::CAP_GSTREAMER);

    // VideoWriter
    cv::VideoWriter irRPIWriter(irSendIRCameraPipe, cv::VideoWriter::fourcc('H', '2', '6', '4'), FRAME_RATE, cv::Size(HORIZONTAL_RESOLUTION, VERTICAL_RESOLUTION), false);
    cv::VideoWriter visibleRPIWriter(visibleSendIRCameraPipe, cv::VideoWriter::fourcc('H', '2', '6', '4'), FRAME_RATE, cv::Size(HORIZONTAL_RESOLUTION, VERTICAL_RESOLUTION), true);

    // cv::VideoWriter debuggerWriter(visibleSendIRCameraPipe, cv::VideoWriter::fourcc('H', '2', '6', '4'), FRAME_RATE, cv::Size(HORIZONTAL_RESOLUTION, VERTICAL_RESOLUTION), true);

    if (!capIR.isOpened() || !visibleCap.isOpened())
    {
        std::cerr << "Failed to open video stream" << std::endl;
        return -1;
    }
    std::cout << "Open video streams" << std::endl;

    // Mat *irFramePtr = &irFrames;
    // Mat *visibleFramePtr = &visibleFrames;
    capIR >> irFrames;
    visibleCap >> visibleFrames;

    std::cout << irFrames.size() << std::endl;
    std::cout << visibleFrames.size() << std::endl;
    imshow("troubleshoot", irFrames);
    imshow("troubleshoot2", visibleFrames);

    waitKey(1);

    cin.get();

    // irFrames = Mat::zeros(Size(HORIZONTAL_RESOLUTION, VERTICAL_RESOLUTION), CV_8UC3);
    // visibleFrames = Mat::zeros(Size(HORIZONTAL_RESOLUTION, VERTICAL_RESOLUTION), CV_8UC3);
    // // Create thread

    // thread t1(captureIRFrames, ref(capIR), irFramePtr, "IR Camera");
    // thread t2(captureVisibleFrames, ref(visibleCap), visibleFramePtr, "Visible Camera");
    // Mat visibleGrayFrame, irGrayFrame;

    // // cv::flip(*irFramePtr, *irFramePtr, 1);

    // while (true)
    // {
    //     irMutex.lock();
    //     visibleMutex.lock();
    //     imshow("IR Camera Frame On Laptop", *irFramePtr);
    //     imshow("Visible Camera Frame On Laptop", *visibleFramePtr);
    //     irMutex.unlock();
    //     visibleMutex.unlock();
    //     // cv::flip(*irFramePtr, *irFramePtr, 1);

    //     // Try to grayscale the frames on the Raspberry PI via terminal before sending them over
    //     cv::cvtColor(*visibleFramePtr, visibleGrayFrame, cv::COLOR_BGR2GRAY);
    //     cv::cvtColor(*irFramePtr, irGrayFrame, cv::COLOR_BGR2GRAY);

    //     // cv::imshow("irGrayFrame", irGrayFrame);
    //     // cv::imshow("visibleGrayFrame", visibleGrayFrame);

    //     cv::Rect roiIR(irGrayFrame.cols / 4, irGrayFrame.rows / 4, irGrayFrame.cols / 2, irGrayFrame.rows / 2);
    //     cv::Rect roiVisible(visibleGrayFrame.cols / 4, visibleGrayFrame.rows / 4, visibleGrayFrame.cols / 2, visibleGrayFrame.rows / 2);

    //     // Zoom in on both visible and IR frames
    //     croppedVisibleFrames = visibleGrayFrame(roiVisible);
    //     croppedIrFrames = irGrayFrame(roiIR);

    //     // Convert the cropped frames to IPL_DEPTH_8U
    //     croppedVisibleFrames.convertTo(croppedVisibleFrames, CV_8U);
    //     croppedIrFrames.convertTo(croppedIrFrames, CV_8U);

    //     // Resize to both perspective frames
    //     cv::resize(croppedVisibleFrames, croppedVisibleFrames, visibleFrames.size());
    //     cv::resize(croppedIrFrames, croppedIrFrames, irFrames.size());

    //     // cv::imshow("croppedVisibleFrames", croppedVisibleFrames);
    //     //  cv::imshow("croppedIrFrames", croppedIrFrames);

    //     // Process the frame using ImgProc_YenThreshold
    //     double foundThresh;
    //     // thresholdimg == binaryMask
    //     yenThresholdedFrame = ImgProc_YenThreshold(croppedIrFrames, false, foundThresh);
    //     remapMin = (int)foundThresh;
    //     // // cv::Mat processedFrame;
    //     int topkvalue = 0;
    //     // // remap_lut_threshold(yenThresholdedFrame, processedFrame, 0.1, foundThresh, topkvalue);
    //     remap_lut_threshold(yenThresholdedFrame, processedFrame, 0.1, remapMin, topkvalue);

    //     ColoredFrame = cv::Ptr<cv::Mat>(new cv::Mat());
    //     // The result will be an RGB
    //     cv::applyColorMap(processedFrame, *ColoredFrame, cv::COLORMAP_JET);

    //     std::vector<cv::Mat> channels;
    //     cv::split(*ColoredFrame, channels);

    //     std::vector<cv::Mat> chansToMerge = {channels[0], channels[1], channels[2], yenThresholdedFrame};
    //     cv::merge(&chansToMerge[0], chansToMerge.size(), *ColoredFrame);

    //     // Projects Visible to IR THIS IS WHAT WE WANT
    //     cv::warpPerspective(croppedVisibleFrames, visibleWarpedFrame, visibleToIRHomography, croppedIrFrames.size());

    //     visibleWarpedFrame.convertTo(visibleWarpedFrame, (*ColoredFrame).type());
    //     cv::cvtColor(visibleWarpedFrame, visibleWarpedFrame, cv::COLOR_GRAY2BGRA);

    //     cv::addWeighted(*ColoredFrame, THRESHOLD_WEIGHT, visibleWarpedFrame, WARPEDFRAME_WEIGHT, 0, visibleToIRProjectedFrame);

    //     // cv::imshow("Heat Map Projected Onto IR", *ColoredFrame);
    //     // cv::imshow("Combined Visible --> IR Warped Overlay With Yen Threshold Overlay  ", visibleToIRProjectedFrame);
    //     // drop alpha
    //     //  bgra --> opencv color order
    //     Mat splitFrames[4];
    //     // split(src,destination)
    //     split(visibleToIRProjectedFrame, splitFrames);
    //     std::vector<cv::Mat> mergeFrames = {
    //         splitFrames[0],
    //         splitFrames[1],
    //         splitFrames[2],
    //     };

    //     Mat mergedFrame;
    //     merge(mergeFrames, mergedFrame);
    //     // std::cout << "mergedFrame depth: " << mergedFrame.depth() << std::endl;
    //     // std::cout << "mergedFrame channels: " << mergedFrame.channels() << std::endl;

    //     // This is used so make sure to uncomment this
    //     // visibleRPIWriter.write(mergedFrame);

    //     //  irRPIWriter.write(croppedIrFrames);
    //     // debuggerWriter.write(visibleToIRProjectedFrame);
    //     if (waitKey(1) == 27)
    //         break;
    // }

    // t1.join();
    // t2.join();

    // Release the VideoCapture objects and close any OpenCV windows
    capIR.release();
    visibleCap.release();
    destroyAllWindows();

    return 0;
}
