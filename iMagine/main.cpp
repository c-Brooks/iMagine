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
#include "learn.hpp"


#define MAX_THRESHOLD_HUE 179
#define MAX_THRESHOLD_SAT 255
#define MAX_THRESHOLD_VAL 255

using namespace cv;
using namespace std;


// -------------------------------------------------------------------------------------------------------
// ----------------------------------------------- iMagine -----------------------------------------------
// -------------------------------------------------------------------------------------------------------


/*  TODO:
 *      When tracking, discourage "jumps" of object -> weighted area  = area / DELTA_P
 *      Fix the bug where trackbars freeze the program
 *      Keep drawing after draw is erased
 */







//--------------------------------------------------------------------------------------------
// --------------------------------------- STOLEN CODE ---------------------------------------
//--------------------------------------------------------------------------------------------



bool plotSupportVectors = true;
int numTrainingPoints = 200;
int numTestPoints = 2000;
int size = 200;
int eq = 0;


// plot data and class
void plot_binary(cv::Mat& data, cv::Mat& classes, string name) {
    cv::Mat plot(size, size, CV_8UC3);
    plot.setTo(cv::Scalar(255.0,255.0,255.0));
    for(int i = 0; i < data.rows; i++) {
        
        float x = data.at<float>(i,0) * size;
        float y = data.at<float>(i,1) * size;
        
        if(classes.at<float>(i, 0) > 0) {
            cv::circle(plot, Point(x,y), 2, CV_RGB(255,0,0),1);
        } else {
            cv::circle(plot, Point(x,y), 2, CV_RGB(0,255,0),1);
        }
    }
    cv::imshow(name, plot);
}



// function to learn
int f(float x, float y, int equation) {
    switch(equation) {
        case 0:
            return y > sin(x*10) ? -1 : 1;
            break;
        case 1:
            return y > cos(x * 10) ? -1 : 1;
            break;
        case 2:
            return y > 2*x ? -1 : 1;
            break;
        case 3:
            return y > tan(x*10) ? -1 : 1;
            break;
        default:
            return y > cos(x*10) ? -1 : 1;
    }
}



// label data with equation
cv::Mat labelData(cv::Mat points, int equation) {
    cv::Mat labels(points.rows, 1, CV_32FC1);
    for(int i = 0; i < points.rows; i++) {
        float x = points.at<float>(i,0);
        float y = points.at<float>(i,1);
        labels.at<float>(i, 0) = f(x, y, equation);
    }
    return labels;
}

//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------



int main(int argc, char** argv)
{
    
    Mat image, edge;
    String command;

    
    
    cout << "\n ----------------------------------------" << endl
         <<   " --------- Welcome to iMagine -----------" << endl
         <<   " ----------------------------------------" << endl;
    
    
    cout << " For help, type 'help' \n" << endl;
    
    
    
    Mat gray, hsv, thresh, imFinal;
    int h_min = 0, h_max = MAX_THRESHOLD_HUE, s_min = 0, s_max = MAX_THRESHOLD_SAT, v_min = 0, v_max = MAX_THRESHOLD_VAL;

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
                
                // TODO: Make trackbars look nicer
                
                cvCreateTrackbar("Min Hue", "Thresholds", &h_min, MAX_THRESHOLD_HUE);
                cvCreateTrackbar("Max Hue", "Thresholds", &h_max, MAX_THRESHOLD_HUE);
                cvCreateTrackbar("Min Sat", "Thresholds", &s_min, MAX_THRESHOLD_SAT);
                cvCreateTrackbar("Max Sat", "Thresholds", &s_max, MAX_THRESHOLD_SAT);
                cvCreateTrackbar("Min Val", "Thresholds", &v_min, MAX_THRESHOLD_VAL);
                cvCreateTrackbar("Max Val", "Thresholds", &v_max, MAX_THRESHOLD_VAL);
                
                inRange(hsv, Scalar(h_min, s_min, v_min), Scalar(h_max, s_max, v_max), thresh);
                
                imshow("Threshold Image", thresh);
                
                if (cvWaitKey(10) == 27) {                   // esc
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
            flip(image, image, 1);
            cvtColor(image, hsv, CV_BGR2HSV);

            inRange(hsv, Scalar(h_min, s_min, v_min), Scalar(h_max, s_max, v_max), thresh);
            
            imFinal = operators::morphOps(thresh);

            findContours(imFinal, countours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
            int numObj = (int) hierarchy.size();
            
            if(numObj > 0)// && numObj < 100) // more than 100 -> noisy image
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

            operators::drawCircle(x, y, areaMax, image);
            imshow("Image", image);
            areaMax = 0.0;
            
            
            if (cvWaitKey(10) == 27) {                   // esc
                cvDestroyAllWindows();
                cvWaitKey(10);
                break;
            }
        }
    }
        
        // DRAW
        //  Uses the tracked object as a pen
        //  Press spacebar to clear lines
        
    else if(!command.compare("draw"))
    {
        cvNamedWindow ("Drawing", CV_WINDOW_AUTOSIZE);
        int newX = -1, newY = -1;
        Mat imgTemp;
        Mat lines = Mat::zeros(image.size(), CV_8UC3);
        
        while(cvWaitKey(10) != 27)
        {
                
            image = cvQueryFrame (capture);
            flip(image, image, 1);
            cvtColor(image, hsv, CV_BGR2HSV);
            
            inRange(hsv, Scalar(h_min, s_min, v_min), Scalar(h_max, s_max, v_max), thresh);
            
            imFinal = operators::morphOps(thresh);
        
            findContours(imFinal, countours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
            int numObj = (int) hierarchy.size();
            
            if(numObj > 0)// && numObj < 100) // more than 100 -> noisy image
            {
                for(int i = 0; i < numObj; i++)
                {
                    moment = moments((Mat)countours[i]);
                    area = moment.m00;
                    
                    // Find the largest area of white pixels
                    if(area > 400  && area > areaMax)
                    {
                        newX = moment.m10/area;
                        newY = moment.m01/area;
                        areaMax = area;
                    }
                }
            }
            if( newX > 0 && newY > 0 && x > 0 && y > 0 )
              // && abs(x-newX) < 200 && abs(y-newY) < 100 ) // To prevent jumps
//               && getch() == 32) // getch is for windows
            
                line(lines, Point(x,y), Point(newX, newY), Scalar(0,0,255), 2); // draw line from old point to new point
            
            // update points
            x = newX;
            y = newY;
            
            
            imgTemp = image + lines;
            imshow("Drawing", imgTemp);
            areaMax = 0.0;
 
            if (cvWaitKey(10) == 32) // spacebar - erase lines
            {
                lines = Mat::zeros(lines.size(), CV_8UC3);
                newX = -1;
                newY = -1;
   
            }
  
        }
        destroyAllWindows();
        waitKey(10);
        }

        // Face Detection using Haar Cascade
        
        else if (!command.compare("face") || !command.compare("faces"))
        {
            cvNamedWindow ("Faces", CV_WINDOW_AUTOSIZE);

            while(cvWaitKey(30) != 27)                  // wait esc for 30ms because more expensive operation
            {
                image = cvQueryFrame (capture);
                flip(image, image, 1);

                image = operators::detectFace(image);
                imshow("Faces", image);
            }
            destroyAllWindows();
            waitKey(10);
        }
        
        else if (!command.compare("train")){

            cv::Mat trainingData(numTrainingPoints, 2, CV_32FC1);
            cv::Mat testData(numTestPoints, 2, CV_32FC1);
            
            cv::randu(trainingData,0,1);
            cv::randu(testData,0,1);
            
            cv::Mat trainingClasses = labelData(trainingData, eq);
            cv::Mat testClasses = labelData(testData, eq);
            
            plot_binary(trainingData, trainingClasses, "Training Data");
            plot_binary(testData, testClasses, "Test Data");

            
            
            waitKey(10);
        }

            else if (!command.compare("close")){
                destroyAllWindows();
                waitKey(10);
            }
            
            else if (!command.compare("bye")){
                destroyAllWindows();
                waitKey(10);
                return 0;
            }
            else
                cout << " Not a valid command.\n For help, type help." << endl;
        }

        destroyAllWindows();
        waitKey(10);
        return 0;
}