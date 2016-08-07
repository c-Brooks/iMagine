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
#include "data.hpp"


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

const int sampleSize = 946;


const char* FILENAME =  "mlp_classifier.yml";
const int classCount = 10;

data trainData;

FileStorage fs;

string file_to_load = "";

CvANN_MLP* mlp = new CvANN_MLP;

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
            
            if(numObj > 0)
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
            
            if(numObj > 0)
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
            if( newX > 0 && newY > 0 && x > 0 && y > 0)
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
                flip(image, image, 1);                  // To look more natural

                // image - the frame captured from cam + circle around the faces
                image = operators::detectFace(image);
                imshow("Faces", image);
            }
            destroyAllWindows();
            waitKey(10);
        }
        
        else if (!command.compare("train") || !command.compare("t")){

            // Create layers of neural net
            Mat layers = Mat(4, 1, CV_32SC1);
            layers.row(0) = cv::Scalar(1024); // size of input
            layers.row(1) = cv::Scalar(10);
            layers.row(2) = cv::Scalar(15);
            layers.row(3) = cv::Scalar(classCount);
            
            // Set parameters
            CvANN_MLP_TrainParams params;
            CvTermCriteria criteria;
            criteria.max_iter = 100;
            criteria.epsilon = 0.00001f;
            criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
            params.train_method = CvANN_MLP_TrainParams::BACKPROP;
            params.bp_dw_scale = 0.05f;
            params.bp_moment_scale = 0.05f;
            params.term_crit = criteria;
            

            mlp->create(layers);
            
            trainData.prepareData();

            mlp->train(trainData.getTrainData(), trainData.getTrainResp(), Mat(), Mat(), params);
        }
        
        
        // Load the classifier from xml file
        // NOTE :  If mlp is initialized, this will overwrite it
        else if (!command.compare("load")){
            fs.open(FILENAME, FileStorage::READ); // read from file storage
            if(fs.isOpened())
                mlp->load(FILENAME);
            fs.release();
        }
        
        // Save the classifier to xml file
        else if (!command.compare("save")){
            fs.open(FILENAME, FileStorage::WRITE); // write to file storage
            if(fs.isOpened())
                mlp->save(FILENAME);
            fs.release();
        }
        
        else if (!command.compare("predict")){
            data testData;
            data trainData;
            
            trainData.prepareData();
            

            Mat  testClasses = trainData.getTrainResp();
            
            mlp->predict( trainData.getTrainData(), testClasses );
            
        }
        
            else if (!command.compare("bye")){
                fs.release();
                destroyAllWindows();
                waitKey(10);
                return 0;
            }
        
        // TEST
        // Split data randomly, train on a portion of the data and then classify the rest.
        // Try and find the one with the smallest error rate
            else if (!command.compare("test")){
                
                
                data testData;
                
                testData.prepareData();
                
                testData.splitData(sampleSize/2);
                
                Mat  prediction = Mat(146, classCount, CV_32F); // Responses for test data
                
                
             //   cout << "testData: " << testData.getTestData() << endl << "TESTDATA" << endl;
               // testData.printData();
                
                
                // Create layers of neural net
                Mat layers = Mat(4, 1, CV_32SC1);
                layers.row(0) = cv::Scalar(1024); // size of input
                layers.row(1) = cv::Scalar(10);
                layers.row(2) = cv::Scalar(15);
                layers.row(3) = cv::Scalar(classCount);
                
                // Set parameters
                CvANN_MLP_TrainParams params;
                CvTermCriteria criteria;
                criteria.max_iter = 100;
                criteria.epsilon = 0.00001f;
                criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
                params.train_method = CvANN_MLP_TrainParams::BACKPROP;
                params.bp_dw_scale = 0.05f;
                params.bp_moment_scale = 0.05f;
                params.term_crit = criteria;
                
                mlp->create(layers);
                
                mlp->train(testData.getTrainData(), testData.getTrainResp(), Mat(), Mat(), params);

                mlp->predict(testData.getTestData(), prediction);
                
                int maxVal = 0, tempResp = 0;
                float accuracy;
                
                // Print out predictions
                for(int i=0; i < prediction.rows; i++)
                {
                    for (int j = 0; j < classCount; j++)
                    {
                        if(prediction.at<float>(i,j) > maxVal){
                            maxVal = prediction.at<float>(i,j);
                            tempResp = j;
                        }
                    //    if(maxVal > 230) break;
                        
                    }
                    
                    cout << tempResp << ", ";

                    // Figure out the certainty of classification
                    
                    cout << (abs(maxVal - 255)) << endl;
                    
                    maxVal = 0;

                }
                
            }
                 
            else
                cout << " Not a valid command.\n For help, type help." << endl;
        
        }
        delete mlp;
        fs.release();
        destroyAllWindows();
        waitKey(10);
        return 0;
}




