//
//  main.cpp
//  iMagine
//
//  Created by Corey Brooks on 2016-06-29.
//  Copyright © 2016 Corey Brooks. All rights reserved.

#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <opencv/ml.h>

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




const int numTrainingPoints = 400;
const int numTestPoints = 1500;


void mlp(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses) {
    
    cv::Mat layers = cv::Mat(4, 1, CV_32SC1);
    
    layers.row(0) = cv::Scalar(2);
    layers.row(1) = cv::Scalar(10);
    layers.row(2) = cv::Scalar(15);
    layers.row(3) = cv::Scalar(1);
    
    CvANN_MLP mlp;
    CvANN_MLP_TrainParams params;
    CvTermCriteria criteria;
    criteria.max_iter = 100;
    criteria.epsilon = 0.00001f;
    criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
    params.train_method = CvANN_MLP_TrainParams::BACKPROP;
    params.bp_dw_scale = 0.05f;
    params.bp_moment_scale = 0.05f;
    params.term_crit = criteria;
    
    mlp.create(layers);
    
    // train
    mlp.train(trainingData, trainingClasses, cv::Mat(), cv::Mat(), params);
    
    cv::Mat response(1, 1, CV_32FC1);
    cv::Mat predicted(testClasses.rows, 1, CV_32F);
    for(int i = 0; i < testData.rows; i++) {
        cv::Mat response(1, 1, CV_32FC1);
        cv::Mat sample = testData.row(i);
        
        mlp.predict(sample, response);
        predicted.at<float>(i,0) = response.at<float>(0,0);
    }
    operators::plot_binary(testData, predicted, "Predictions");
}




// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------



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

        
        // SET
        //  Sets the threshold values for HSV image
        //  Once they are set, object can be tracked with track command
        
        if(!command.compare("set"))
        {
            CvCapture *capture = cvCreateCameraCapture(0);
            cvStartWindowThread();
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
        CvCapture *capture = cvCreateCameraCapture(0);
        cvStartWindowThread();
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
        CvCapture *capture = cvCreateCameraCapture(0);
        cvStartWindowThread();
        
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
            CvCapture *capture = cvCreateCameraCapture(0);
            cvStartWindowThread();
            
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
        
        else if (!command.compare("train") || !command.compare("t")){

            Mat trainingData(numTrainingPoints, 2, CV_32FC1);
            Mat testData(numTestPoints, 2, CV_32FC1);

            randu(trainingData,0,1);
            randu(testData,0,1);
            
            Mat trainingClasses = learn::labelData(trainingData);
            Mat testClasses     = learn::labelData(testData);
            
            operators::plot_binary(trainingData, trainingClasses, "Training Data");
            
            operators::plot_binary(testData, testClasses, "Test Data");

            mlp(trainingData, trainingClasses, testData, testClasses);
            
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