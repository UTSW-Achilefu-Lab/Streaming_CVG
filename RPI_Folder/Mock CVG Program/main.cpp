#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "yen_threshold.h"

#define NOIR_CAMERA 0
#define VISIBLE_CAMERA 1
#define CHESSBOARD_COLUMNS 9
#define CHESSBOARD_ROWS 6
#define LINE_THICKNESS 1
#define NUMBER_OF_CALIBRATION_IMAGES 1
#define CALIBRATION_DELAY 1000 // In milliseconds
#define THRESHOLD_WEIGHT 0.4   // Increase to see more of the Yen Threshold Image
#define WARPEDFRAME_WEIGHT 0.6
#define ESC_KEY 27

using namespace cv;

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

int main()
{
    // Path to the YAML file
    std::string filename = "/home/pi/Onboard_VS_Streaming_RPI/Archive Folder/Homography/build/homography_matrix.yaml";
    std::cout << "Opening file: " << filename << std::endl;

    // Open the file using FileStorage
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "Failed to open " << filename << std::endl;
        return -1;
    }

    // Variables to store the matrices
    cv::Mat infraredToVisibleHomography, visibleToInfraredHomography;
    cv::Ptr<cv::Mat> ColoredFrame;
    // Read the matrices from the file
    fs["infraredToVisibleHomography"] >> infraredToVisibleHomography;
    fs["visibleToIRHomography"] >> visibleToInfraredHomography;

    // Close the file
    fs.release();

    // Check if matrices are empty after reading from file
    if (infraredToVisibleHomography.empty() || visibleToInfraredHomography.empty())
    {
        std::cerr << "Failed to read matrices from file" << std::endl;
        return -1;
    }

    // Output the read values
    std::cout << "infraredToVisibleHomography:\n"
              << infraredToVisibleHomography << std::endl;
    std::cout << "visibleToInfraredHomography:\n"
              << visibleToInfraredHomography << std::endl;

    // Open two camera streams
    cv::VideoCapture capIR(NOIR_CAMERA);
    cv::VideoCapture capVisible(VISIBLE_CAMERA);

    capIR.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('B', 'G', '1', '0'));
    capVisible.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('B', 'G', '1', '0'));

    capIR.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capIR.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capVisible.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capVisible.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // Check if the cameras are opened
    if (!capIR.isOpened() || !capVisible.isOpened())
    {
        std::cerr << "Error: Cameras not accessible" << std::endl;
        return -1;
    }

    printf("\nPress ESC key to take image...\r\n");
    bool previewFlag = true;
    cv::Mat irFrames, visibleFrames, combinedFrames, croppedVisibleFrames, croppedIrFrames, irWarpedFrame, visibleWarpedFrame, successfulCalibrationFrameCaptured, warpedFramesSideBySide, zoomedIrFramesGray, visibleFramesGray, visibleToIRProjectedFrame, rawCombinedFrames, visibleFramesBGR, processedFrame, yenThresholdedFrame, combinedZoomedFrames;
    double foundThresh = 0.0;
    int topkvalue;
    int remapMin = 0;

    // ------------------------ [ VERIFY HOMOGRAPHY VALUES AND YEN THRESHOLDING ] ------------------------ //

    while (true)
    {
        // Mat visibleFrames, irFrames;
        capVisible >> visibleFrames;
        capIR >> irFrames;
        cv::flip(irFrames, irFrames, 1);

        // Check if any of the frames is empty
        if (visibleFrames.empty() || irFrames.empty())
        {
            std::cerr << "Error: Unable to capture frame\n";
            break;
        }

        // If you want to use hconcat() as a sanity check instead of viewing each frame individually
        // The following three lines of code must be commented out since visibleFrame is being grayscaled from a data type of 16 --> 0
        // Couldn't figure out how to convert the visibleFrames to 'fit' with the irFrames that has a data type of 16

        // cvtColor that's doing the  demosaicing expects a single-channel input image to produce a 3 or 4 channel output
        // So grayscaling it first is nessessary
        cvtColor(visibleFrames, visibleFrames, cv::COLOR_BGR2GRAY);
        cvtColor(visibleFrames, visibleFramesBGR, cv::COLOR_BayerBG2BGR);

        // cv::resize(visibleFrames, visibleFrames, irFrames.size());
        // visibleFrames.convertTo(visibleFrames, irFrames.type());

        // Define the region of interest (ROI) for zooming in (half size)
        cv::Rect roiVisible(visibleFramesBGR.cols / 4, visibleFramesBGR.rows / 4, visibleFramesBGR.cols / 2, visibleFrames.rows / 2);
        cv::Rect roiIR(irFrames.cols / 4, irFrames.rows / 4, irFrames.cols / 2, irFrames.rows / 2);

        // Zoom in on both visible and IR frames
        croppedVisibleFrames = visibleFrames(roiVisible);
        croppedIrFrames = irFrames(roiIR);

        // Resize to both perspective frames
        cv::resize(croppedVisibleFrames, croppedVisibleFrames, visibleFrames.size());
        cv::resize(croppedIrFrames, croppedIrFrames, irFrames.size());

        // Process the frame using ImgProc_YenThreshold
        double foundThresh;
        // thresholdimg == binaryMask
        yenThresholdedFrame = ImgProc_YenThreshold(croppedIrFrames, false, foundThresh);
        // std::cout << yenThresholdedFrame.channels() << std::endl;
        // std::cin.get();
        remapMin = (int)foundThresh;
        // cv::Mat processedFrame;
        int topkvalue = 0;
        // remap_lut_threshold(yenThresholdedFrame, processedFrame, 0.1, foundThresh, topkvalue);
        remap_lut_threshold(yenThresholdedFrame, processedFrame, 0.1, remapMin, topkvalue);

        // std::cout << processedFrame.channels() << std::endl;

        ColoredFrame = cv::Ptr<cv::Mat>(new cv::Mat());
        // The result will be an RGB
        cv::applyColorMap(processedFrame, *ColoredFrame, cv::COLORMAP_JET);

        std::vector<cv::Mat> channels;
        cv::split(*ColoredFrame, channels);

        std::vector<cv::Mat> chansToMerge = {channels[0], channels[1], channels[2], yenThresholdedFrame};
        // std::cout << channels[0].channels() << channels[1].channels() << channels[2].channels();
        cv::merge(&chansToMerge[0], chansToMerge.size(), *ColoredFrame);

        // Projects IR to VISIBLE
        // void cv::warpPerspective(cv::InputArray src, cv::OutputArray dst, cv::InputArray M, cv::Size dsize, int flags = 1, int borderMode = 0, const cv::Scalar &borderValue = cv::Scalar())
        // cv::warpPerspective(croppedIrFrames, irWarpedFrame, infraredToVisibleHomography, croppedVisibleFrames.size());

        // Projects Visible to IR THIS IS WHAT WE WANT
        cv::warpPerspective(croppedVisibleFrames, visibleWarpedFrame, visibleToInfraredHomography, croppedIrFrames.size());

        // Uncomment if you want to see the size/resolution and data types of the frames that are being passed to debugOutput()
        // debugOutput(visibleFrames, irFrames, croppedIrFrames, croppedVisibleFrames, irWarpedFrame, visibleWarpedFrame, yenThresholdedFrame);

        // ORIGINAL
        // void cv::addWeighted(cv::InputArray src1, double alpha, cv::InputArray src2, double beta, double gamma, cv::OutputArray dst, int dtype = -1)
        // cv::addWeighted(yenThresholdedFrame, THRESHOLD_WEIGHT, visibleWarpedFrame, WARPEDFRAME_WEIGHT, 0, visibleToIRProjectedFrame);

        // cv::resize(*ColoredFrame, *ColoredFrame, visibleWarpedFrame.size());

        visibleWarpedFrame.convertTo(visibleWarpedFrame, (*ColoredFrame).type());
        cv::cvtColor(visibleWarpedFrame, visibleWarpedFrame, cv::COLOR_GRAY2BGRA);

        cv::addWeighted(*ColoredFrame, THRESHOLD_WEIGHT, visibleWarpedFrame, WARPEDFRAME_WEIGHT, 0, visibleToIRProjectedFrame);

        // void cv::putText(cv::InputOutputArray img, const cv::String &text, cv::Point org, int fontFace, double fontScale, cv::Scalar color, int thickness = 1, int lineType = 8, bool bottomLeftOrigin = false)

        //  Concatenate  frames won't work since visibleFrames is being grayscaled
        //  To use hconcat() those lines of code following the grayscaling of visibleFrames must be commented out 2/24/24
        // cv::hconcat(irWarpedFrame, visibleWarpedFrame, combinedWarpedFrames);

        // Needed to show raw frames together
        cvtColor(visibleFrames, visibleFramesBGR, cv::COLOR_GRAY2BGR);
        cv::hconcat(irFrames, visibleFramesBGR, rawCombinedFrames);
        // cv::hconcat(croppedIrFrames, croppedVisibleFrames, combinedZoomedFrames);

        // cv::imshow("Zoomed Homography camera feed ", combinedZoomedFrames);
        //  cv::imshow("Warped Frames | IR --> Visible | Visible --> IR", combinedWarpedFrames);
        //  cv::imshow("yenThresholdedFrame ", yenThresholdedFrame);

        // Individually using imshow() to display frames seperately works fine
        // Projects IR to VISIBLE
        // cv::imshow("irWarpedFrame Frame", irWarpedFrame);

        // Projects Visible to IR, this is what's implemented in CVG
        // cv::imshow("visibleWarpedFrame Frame", visibleWarpedFrame);

        // Display frames individually
        // cv::imshow("Visible Frame 'Raw' ", visibleFrames);
        // cv::imshow("IR Frame 'Raw' ", irFrames);

        //cv::imshow("Raw camera feed", rawCombinedFrames);
        //cv::imshow("croppedIrFrames", croppedIrFrames);
        //cv::imshow("croppedVisibleFrames", croppedVisibleFrames);

        //cv::imshow("visibleWarpedFrame", visibleWarpedFrame);
        cv::imshow("Combined Visible --> IR Warped Overlay With Yen Threshold Overlay  ", visibleToIRProjectedFrame);
        //cv::imshow("yenThresholdedFrame ", yenThresholdedFrame);
        //cv::imshow("Heat Map Projected Onto IR", *ColoredFrame);

        //  Pres ESC key to exit program
        if (cv::waitKey(1) == ESC_KEY)
        {
            std::cout << "Exiting...\n";
            break;
        }
    }

    cv::destroyAllWindows();

    cv::imwrite("Raw_Visible_Frame.JPG", visibleFrames);
    cv::imwrite("Raw_IR_Frame.JPG", irFrames);
    cv::imwrite("Cropped_Visible_Frame.JPG", croppedVisibleFrames);
    cv::imwrite("Cropped_IR_Frame.JPG", croppedIrFrames);
    cv::imwrite("Warped_Visible_To_IR.JPG", visibleWarpedFrame);
    cv::imwrite("Warped_IR_To_Visible.JPG", irWarpedFrame);
    cv::imwrite("YenThreshold_On_IR_Frame.JPG", yenThresholdedFrame);
    cv::imwrite("ColoredFrame.JPG", *ColoredFrame);
    cv::imwrite("visibleToIRProjectedFrame.JPG", visibleToIRProjectedFrame);
    // cv::imwrite("coloredVisible.JPG", visibleFrames);

    return 0;
}
