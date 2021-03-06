//
//  operators.hpp
//  iMagine
//
//  Created by Corey on 2016-06-29.
//  Copyright © 2016 Corey. All rights reserved.
//

#ifndef operators_hpp
#define operators_hpp

#include <stdio.h>
#include <opencv/cv.h>


using namespace std;
using namespace cv;


class operators {
private:
    static Mat grey, sobelIm;    
public:
    static Mat  edgeDetect(Mat src);
    static Mat  gblur(Mat src, int kernLength);
    static void drawCircle(int x, int y, double area, Mat &frame);
    static void help();
    static Mat  morphOps(Mat src);
    static Mat  detectFace(Mat src);
    static void plot_binary(Mat data, Mat classes, string name);
    static Mat data2plot(Mat data, string name);
    static Mat plot2data(Mat data, string name);

    
};
#endif // End operators.h
