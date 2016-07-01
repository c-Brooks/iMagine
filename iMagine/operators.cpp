
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

void operators::help() {
    cout << "\n     COMMANDS:" << endl
    << " help ...................... Opens help menu " << endl
    << " cam ....................... Edge detects feed from webcam " << endl
    << " open ...................... Opens loaded image " << endl
    << " new ....................... Load a new image " << endl
    << " edge ...................... Find edges with Canny operator " << endl
    << " gblur ..................... Gaussian blur " << endl
    //		<< " save ...................... Save the current image" << endl
    << " bye ....................... Exit imProc" << endl
    
    << "\n iMagine is created by Corey Brooks, UBC computer engineering." << endl
    << "\n\n";
}

const string keys =
"{                   |                   |                   }"
"{ help h about ?    |                   |                   }"




;