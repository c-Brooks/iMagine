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

vector<Mat> data::getTrainData(){
    return trainData;
}

vector<Mat> data::getTrainResp(){
    return trainResp;
}

Mat data::dataAt(int i){
    return trainData.at(i);
}

Mat data::respAt(int i){
    return trainResp.at(i);
}



// SETTER (does data and responses)
// Turns a single file into a vector of matrices
// This file contains 300 samples of 32x32 binary images of digits

void data::prepareData(){ // 946
    
    // Declare datasets
    vector<float>            dataRow;   // 0, 1, 2 ... 9
    dataRow.resize(1024);
    vector<float>            respRow;
    respRow.resize(10);

    Mat dataMat;// = Mat(30, 1024, CV_32F);
    Mat respMat;// = Mat(30, 10, CV_32F);
    
    char valIn;
    
    // Read from file 946
    ifstream readFile ("optdigits-orig.cv");
    
    if(readFile.is_open())
    {
        for(int sampleCount = 0; sampleCount < 30; sampleCount++)
        {
            for(int i = 0; i < 1024; i++){            //  32x32 = 1024
                readFile >> valIn;
                dataRow.at(i) = ((float)(valIn - '0'))*255.0;
                cout << valIn;
            }
            cout << endl;
            
            // Placing matrix into corresponding class
            readFile >> valIn;
            cout << sampleCount << ", " << valIn << endl;
            respRow.at(valIn - '0') = 255.; // minus '0' converts from ascii code to int
            
            
            
    //        dataMat.row(sampleCount) = dataRow;
//            dataRow.copyTo
 //           respMat.push_back(respRow);
        }
    }
    else cout << "test failed" << endl;
    readFile.close();
    
//    data::trainData = dataMat;
//    data::trainResp = respMat;
}










