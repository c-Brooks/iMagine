
#include "operators.hpp"
#include <opencv/cv.h>


using namespace cv;
using namespace std;


// Load Face cascade (.xml file)
CascadeClassifier face_cascade( "/Users/corey/Downloads/frontalFace10/haarcascade_frontalface_alt_tree.xml" );

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

// Morphological operations to prepare binary input image for tracking
// Uses dilate and erode supplied by openCV
Mat operators::morphOps(Mat src){
    
    Mat erodeElem  = getStructuringElement(MORPH_RECT, Size(3,3));
    Mat dilateElem = getStructuringElement(MORPH_RECT, Size(8,8));
    
    dilate(src, src, dilateElem);
    erode(src, src, erodeElem);
    
    erode(src, src, erodeElem);
    dilate(src, src, dilateElem);
    
    return src;
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


// Face detection using a prebuilt xml cascade file
// Returns the image with elipses superimposed over the faces.
Mat operators::detectFace(Mat image)
{
    // Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale( image, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    
    // Draw circles on the detected faces
    for( int i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        ellipse( image, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
    }
    return image;
}
