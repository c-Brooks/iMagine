//
//  data.cpp
//  iMagine
//
//  Created by Corey on 25 07, 2016.
//  Copyright © 2016 Corey. All rights reserved.
//

#include "data.hpp"

#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <opencv/ml.h>

#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;



// GETTERS

Mat data::getTrainData(){
    return trainData;
}

Mat data::getTrainResp(){
    return trainResp;
}

Mat data::getTestData(){
    return testData;
}

Mat data::getTestResp(){
    return testResp;
}


// SETTER (sets data and responses)
// Turns a single file into a vector of matrices
// This file contains 300 samples of 32x32 binary images of digits

void data::prepareData(){ // 946
    
    // Declare datasets
    vector<float>            dataRow; // Vector representation of 32x32 image
    dataRow.resize(1024);
    vector<float>            respRow; // 0, 1, 2 ... 9
    respRow.resize(10);

    Mat dataMat = Mat(sampleSize, 1024, CV_32F);
    Mat respMat = Mat(sampleSize, 10, CV_32F);
    
    char valIn;
    
    // Read from file
    ifstream readFile ("optdigits-orig.cv");
    
    if(readFile.is_open())
    {
        for(int sampleCount = 0; sampleCount < sampleSize; sampleCount++)
        {
            for(int i = 0; i < 1024; i++){            //  32x32 = 1024
                readFile >> valIn;
                dataMat.at<float>(sampleCount, i) = ((float)(valIn - '0'))*255.0;
                cout << valIn;
            }
            cout << endl;
            
            // Placing matrix into corresponding class
            readFile >> valIn;
            cout << sampleCount << ", " << valIn << endl;
            respMat.at<float>(sampleCount, (valIn - '0')) = 255.; // minus '0' converts from ascii code to int
            
        }
    }
    else cout << "test failed" << endl;
    readFile.close();
    
    data::trainData = dataMat;
    data::trainResp = respMat;
}

//  Splits data for testing accuracy
// Assumes there all samples are in data::trainData


void data::splitData(int fulcrum){
    Mat trainDataBuf  = Mat(800, 1024, CV_32F);     // buffer for training data vectors
    Mat testDataBuf   = Mat(146, 1024, CV_32F);     // buffer for testing data vectors
    Mat trainRespBuf  = Mat(800, 10, CV_32F);       // buffer for training response vectors
    Mat testRespBuf   = Mat(146, 10, CV_32F);       // buffer for testing response vectors

    Mat newTrainData = trainData;
    Mat newTrainResp = trainResp;

    for(int i = 0; i < 800; i++){
        if(i < 800){
            trainDataBuf.row(i).copyTo(newTrainData.row(i));
            trainRespBuf.row(i).copyTo(newTrainResp.row(i));
        }
        else {
            testDataBuf.row(i-800).copyTo(testData.row(i));
            testRespBuf.row(i-800).copyTo(testResp.row(i));
        }
  
 }
}








