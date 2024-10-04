#include "yen_threshold.h"

// from https://github.com/aivakov/OpenCV-hacks-C-/blob/master/Yen.cpp
// licensed under GNU General Public License v3.0
using namespace cv;

int Yen(Mat data)
{
    // Ported to C++ by Alexander Ivakov from Java implementation of ImageJ plugin Auto_Threshold

    // Implements Yen  thresholding method
    // 1) Yen J.C., Chang F.J., and Chang S. (1995) "A New Criterion
    //    for Automatic Multilevel Thresholding" IEEE Trans. on Image
    //    Processing, 4(3): 370-378
    // 2) Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
    //    Techniques and Quantitative Performance Evaluation" Journal of
    //    Electronic Imaging, 13(1): 146-165
    //    http://citeseer.ist.psu.edu/sezgin04survey.html
    //
    // M. Emre Celebi
    // 06.15.2007
    // Ported to ImageJ plugin by G.Landini from E Celebi's fourier_0.8 routines

    int total = 0;
    for (int ih = 0; ih < 256; ih++)
    {
        float extr = data.at<float>(ih);
        ;
        total += (int)extr;
    }

    // normalized histogram
    double norm_histo[256];
    for (int ih = 0; ih < 256; ih++)
    {
        norm_histo[ih] = (double)data.at<float>(ih) / total;
    }

    // cumulative normalized histogram
    double P1[256];
    P1[0] = norm_histo[0];
    for (int ih = 1; ih < 256; ih++)
    {
        P1[ih] = P1[ih - 1] + norm_histo[ih];
    }

    double P1_sq[256];
    P1_sq[0] = norm_histo[0] * norm_histo[0];
    for (int ih = 1; ih < 256; ih++)
    {
        P1_sq[ih] = P1_sq[ih - 1] + norm_histo[ih] * norm_histo[ih];
    }

    double P2_sq[256];
    P2_sq[255] = 0.0;
    for (int ih = 254; ih >= 0; ih--)
    {
        P2_sq[ih] = P2_sq[ih + 1] + norm_histo[ih + 1] * norm_histo[ih + 1];
    }

    // Find the threshold that maximizes the criterion
    int threshold = -1;
    double max_crit = std::numeric_limits<double>::min();

    for (int it = 0; it < 256; it++)
    {
        double crit = -1.0 * ((P1_sq[it] * P2_sq[it]) > 0.0 ? log(P1_sq[it] * P2_sq[it]) : 0.0) + 2 * ((P1[it] * (1.0 - P1[it])) > 0.0 ? log(P1[it] * (1.0 - P1[it])) : 0.0);
        if (crit > max_crit)
        {
            max_crit = crit;
            threshold = it;
        }
    }
    return threshold;
}