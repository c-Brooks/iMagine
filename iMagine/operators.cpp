
#include "operators.hpp"
#include <opencv/cv.h>

using namespace cv;
using namespace std;

// Edge detection using Canny operator
Mat operators::edgeDetect(Mat src)
{
    Mat draw, grey, edge;
    
    cvtColor(src, grey, CV_BGR2GRAY);
    
    Canny(grey, edge, 50, 100, 3);
    
    edge.convertTo(draw, CV_8U);
    return draw;
}


// Gaussian Blur
Mat operators::gblur(Mat src, int kernLength)
{
    Mat dst;
    for (int i = 1; i < kernLength; i = i + 2)
    {
        GaussianBlur(src, dst, Size(i, i), 0, 0);
    }
    return dst;
}

// Displays help menu
void operators::help() {
    cout << "\n     COMMANDS:" << endl
    << " help ...................... Opens help menu " << endl
    << " set ....................... Set hue, saturation, and value for object detection. " << endl
    << " track ..................... Tracks an object on webcam feed based on set. " << endl
    << " draw ...................... Draws from the tracked object. Spacebar to clear drawing. " << endl
    << " bye ....................... Exit iMagine" << endl
    
    << "\n iMagine is created by Corey Brooks, UBC computer engineering." << endl
    << "\n\n";
}

// Used for tracking objects
// Draws a circle and coordinates
void operators::drawCircle(int x, int y, double area, Mat &frame)
{
    // Get radius of circle from area
    int radius = sqrt(area) * 0.5624; // r = sqrt(area/pi)
    
    // Draw the tracker
    putText(frame, "["+to_string(x)+", "+to_string(y)+"]", Point(x,y), 1, 1, Scalar(0,255,0));
    circle(frame, Point(x,y), radius, Scalar(0,0,255));
}





























