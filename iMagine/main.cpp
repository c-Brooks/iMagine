//
//  main.cpp
//  iMagine
//
//  Created by Corey Brooks on 2016-06-29.
//  Copyright © 2016 Corey Brooks. All rights reserved.

#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <iostream>

#include "operators.hpp"

#define MAX_THRESHOLD 255

using namespace cv;
using namespace std;


// -------------------------------------------------------------------------------------------------------
// ----------------------------------------------- iMagine -----------------------------------------------
// -------------------------------------------------------------------------------------------------------


int main(int argc, char** argv)
{
    
    Mat image, edge;
    String command;
    
    if (argc != 2)
    {
        cerr << "\n\n Error: no argument specified" << endl;
        return -1;
    }
    
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    
    if (!image.data)
    {
        cerr << "\n  Could not open or find the image" << endl;
        return -1;
    }
    else cout << " Succesfully loaded " << argv[1] << endl;
    
    cout << "\n ----------------------------------------" << endl
    <<   " --------- Welcome to iMagine -----------" << endl
    <<   " ----------------------------------------" << endl;
    
    
    cout << " For help, type 'help' \n" << endl;
    
    
    
    Mat gray, hsv, thresh, imFinal;
    int h_min = 0, h_max = 179, s_min = 0, s_max = MAX_THRESHOLD, v_min = 0, v_max = MAX_THRESHOLD;
    Mat erodeElem  = getStructuringElement(MORPH_RECT, Size(3,3));
    Mat dilateElem = getStructuringElement(MORPH_RECT, Size(8,8));
    
    vector<vector<Point>> countours;
    vector<Vec4i> hierarchy;
    Moments moment;
    double area, areaMax = 0.0;
    int x = 0, y = 0;
    
    while (cin >> command)
    {
        CvCapture *capture = cvCreateCameraCapture(0);
        cvStartWindowThread();
        
        // SET
        //  Sets the threshold values for HSV image
        //  Once they are set, object can be tracked with track command
        
        if(!command.compare("set"))
        {
            cvNamedWindow ("Thresholds", CV_WINDOW_AUTOSIZE);
            cvNamedWindow ("Threshold Image", CV_WINDOW_AUTOSIZE);
            
            while (1)
            {
                image = cvQueryFrame (capture);
                cvtColor(image, hsv, CV_BGR2HSV);
                
                cvCreateTrackbar("Min Hue", "Thresholds", &h_min, 179);
                cvCreateTrackbar("Max Hue", "Thresholds", &h_max, 179);
                cvCreateTrackbar("Min Sat", "Thresholds", &s_min, MAX_THRESHOLD);
                cvCreateTrackbar("Max Sat", "Thresholds", &s_max, MAX_THRESHOLD);
                cvCreateTrackbar("Min Val", "Thresholds", &v_min, MAX_THRESHOLD);
                cvCreateTrackbar("Max Val", "Thresholds", &v_max, MAX_THRESHOLD);
                
                inRange(hsv, Scalar(h_min, s_min, v_min), Scalar(h_max, s_max, v_max), thresh);
                
                imshow("Threshold Image", thresh);
                
                if (cvWaitKey(30) == 27) {                   // esc
                    cvDestroyAllWindows();
                    cvWaitKey(10);
                    break;
                }
            }
        }
        
        // TRACK
        //  After setting HSV, the image is tracked by finding the largest area of whitespace
        
    else if(!command.compare("track"))
    {
        cvNamedWindow ("Image", CV_WINDOW_AUTOSIZE);
        
        while(1)
        {
            image = cvQueryFrame (capture);
            cvtColor(image, hsv, CV_BGR2HSV);

            inRange(hsv, Scalar(h_min, s_min, v_min), Scalar(h_max, s_max, v_max), thresh);
            
            dilate(thresh, imFinal, dilateElem);
            erode(imFinal, imFinal, erodeElem);
            
            erode(imFinal, imFinal, erodeElem);
            dilate(imFinal, imFinal, dilateElem);
            
            findContours(imFinal, countours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
            int numObj = (int) hierarchy.size();
            
            if(numObj > 0 && numObj < 200) // more than 200 -> noisy image
            {
                for(int i = 0; i < numObj; i++)
                {
                    moment = moments((Mat)countours[i]);
                    area = moment.m00;
                    
                    if(area > 400  && area > areaMax)
                    {
                        x = moment.m10/area;
                        y = moment.m01/area;
                        areaMax = area;
                    }
                }
            }
            cout << "Centroid at " << x << ", " << y << endl;
            
            operators::drawCircle(x, y, areaMax, image);
            imshow("Image", image);
            areaMax = 0.0;
            
            
            if (cvWaitKey(30) == 27) {                   // esc
                cvDestroyAllWindows();
                cvWaitKey(10);
                break;
            }
        }
    }

            else if (!command.compare("close")){
                destroyAllWindows();
            }
            
            else if (!command.compare("bye")){
                destroyAllWindows();
                return 0;
            }
            else
                cout << " Not a valid command.\n For help, type help." << endl;
        }

        destroyAllWindows();
        waitKey(10);
        return 0;
}