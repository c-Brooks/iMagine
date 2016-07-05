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
    int h_min, h_max, s_min, s_max, v_min, v_max;
    
    
    vector<vector<Point>> countours;
    vector<Vec4i> hierarchy;
    Moments moment;
    double area, areaMax = 0.0;
    int x = 0, y = 0;
    
    while (cin >> command)
    {
        
        if(!command.compare("cam"))
        {
            cvNamedWindow ("HSV", CV_WINDOW_AUTOSIZE);
            cvNamedWindow ("Thresholds", CV_WINDOW_AUTOSIZE);
            cvNamedWindow ("Threshold Image", CV_WINDOW_AUTOSIZE);
            
            CvCapture *capture = cvCreateCameraCapture(0);
            cvStartWindowThread();
            
            while (1)
            {
                image = cvQueryFrame (capture);
                cvtColor(image, hsv, CV_BGR2HSV);
                //   edge = operators::edgeDetect(operators::gblur(hsv, 7));
                
                //                imshow("HSV", hsv);
                
                cvCreateTrackbar("Min Hue", "Thresholds", &h_min, 179);
                cvCreateTrackbar("Max Hue", "Thresholds", &h_max, 179);
                cvCreateTrackbar("Min Sat", "Thresholds", &s_min, MAX_THRESHOLD);
                cvCreateTrackbar("Max Sat", "Thresholds", &s_max, MAX_THRESHOLD);
                cvCreateTrackbar("Min Val", "Thresholds", &v_min, MAX_THRESHOLD);
                cvCreateTrackbar("Max Val", "Thresholds", &v_max, MAX_THRESHOLD);
                
                inRange(hsv, Scalar(h_min, s_min, v_min), Scalar(h_max, s_max, v_max), thresh);
                
                // TODO: Figure out which kernel to use ??
                
                Mat erodeElem  = getStructuringElement(MORPH_RECT, Size(3,3));
                Mat dilateElem = getStructuringElement(MORPH_RECT, Size(8,8));
                
               
                
                dilate(thresh, imFinal, dilateElem);
                erode(imFinal, imFinal, erodeElem);
                
                erode(imFinal, imFinal, erodeElem);
                dilate(imFinal, imFinal, dilateElem);
                
                findContours(imFinal, countours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
                int numObj = (int) hierarchy.size();
                
                if(numObj > 0 && numObj < 15) // more than 15 -> noisy image
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
                    cout << "Centroid at " << x << ", " << y << endl;
                }
                operators::drawCircle(x, y, thresh);
                
                imshow("Threshold Image", thresh);
                
                
                if (cvWaitKey(30) == 27) {                   // esc
                    cvDestroyAllWindows();
                    cvWaitKey(10);
                    break;
                }
            }
        }
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        /*
        
        
        // TODO: green light near webcam stays lit after exited out of cam (?)
        
        if (!command.compare("open"))               // open loaded image
        {
            cvStartWindowThread();
            namedWindow(argv[1], CV_WINDOW_NORMAL);
            imshow(argv[1], image);				    // open image in separate window
            if(waitKey(0)==27){					// window stays up until esc key is pressed
                destroyWindow(argv[1]);
                waitKey(10);
            }
        }
        
        else if (!command.compare("help"))          // Display help messages
            operators::help();
        
        else if (!command.compare("new"))
            cout << " Function not yet implemented" << endl;
        
        else if (!command.compare("gblur"))         // Gaussian blur
        {
            int kern_lth = 16;
            cvStartWindowThread();
            
            Mat gaussIm = operators::gblur(image, kern_lth);
            
            namedWindow("Original", CV_WINDOW_NORMAL);
            imshow("Original", image);
            
            namedWindow("Blurred", CV_WINDOW_NORMAL);
            imshow("Blurred", gaussIm);				 // open image in separate window
            
            if(waitKey(0)==27){					// window stays up until esc key is pressed
                destroyWindow("Original");
                destroyWindow("Blurred");
                waitKey(10);
            }
        }
        
        else if (!command.compare("edge"))           // Canny edge detection
        {
            Mat edgeIm = operators::edgeDetect(operators::gblur(image, 7));
            
            namedWindow("Original", CV_WINDOW_NORMAL);
            imshow("Original", image);
            
            namedWindow("Edge", CV_WINDOW_NORMAL);
            imshow("Edge", edgeIm);
            
            if(waitKey(0)==27){                     // window stays up until esc key is pressed
                destroyWindow("Original");
                destroyWindow("Edge");
                waitKey(10);
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
         */
        destroyAllWindows();
        waitKey(10);
        return 0;
    }
}