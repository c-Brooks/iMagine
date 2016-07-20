//
//  learn.cpp
//  iMagine
//
//  Created by Corey on 15 07, 2016.
//  Copyright © 2016 Corey. All rights reserved.
//

#include "learn.hpp"

#include <opencv2/ml.hpp>
#include <opencv2/core.hpp>

#include <cstdio>
#include <vector>
#include <iostream>


using namespace cv;
using namespace std;

// This is a class to train and classify numbers
// The learning method is Multi-Layered Perceptron, a method that is very effective for text recognition.

/*

CvANN_MLP_TrainParams::CvANN_MLP_TrainParams()
{
    term_crit = cvTermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.01 );
    train_method = RPROP;
    bp_dw_scale = bp_moment_scale = 0.1;
    rp_dw0 = 0.1; rp_dw_plus = 1.2; rp_dw_minus = 0.5;
    rp_dw_min = FLT_EPSILON; rp_dw_max = 50.;
}

void learn::train(){ // const string& filename){
    
}




// Trains the MLP with images from training vector
// The training images are 80% of all sample images, test images are the other 20%
void learn::createClassifier(const string& data_filename, const string& from_file, const string& to_file){
    
    const int classCount = 10; // 0 .. 9

    bool plotSupportVectors = true;
    int numTrainingPoints = 200;
    int numTestPoints = 2000;
    int size = 200;
    int eq = 0;

    
    
    

    cv::Mat layers = cv::Mat(4, 1, CV_32SC1);
    
    layers.row(0) = cv::Scalar(2);
    layers.row(1) = cv::Scalar(10);
    layers.row(2) = cv::Scalar(15);
    layers.row(3) = cv::Scalar(1);
    
    

    CvANN_MLP mlp = CvANN_MLP(); // = CvStatModel::load( data_filename );
    CvANN_MLP_TrainParams params = CvANN_MLP_TrainParams();
    
    CvTermCriteria criteria;
    criteria.max_iter = 100;
    criteria.epsilon = 0.00001f;
    criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
    
    params.train_method = CvANN_MLP_TrainParams::BACKPROP;
    params.bp_dw_scale = 0.05f;
    params.bp_moment_scale = 0.05f;
    params.term_crit = criteria;
    
    mlp.create(layers);
    

    
    
    cv::Mat response(1, 1, CV_32FC1);
    cv::Mat predicted(testClasses.rows, 1, CV_32F);
    for(int i = 0; i < testData.rows; i++) {
        cv::Mat response(1, 1, CV_32FC1);
        cv::Mat sample = testData.row(i);
        
        mlp.predict(sample, response);
        predicted.at<float>(i,0) = response.at<float>(0,0);
        
    }
    
    cout << "Accuracy_{MLP} = " << evaluate(predicted, testClasses) << endl;
    plot_binary(testData, predicted, "Predictions Backpropagation");
}
    */
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
//}

/*
static CvANN_MLP load_classifier(const string& filename_to_load)
{
    // load classifier from the specified file
    CvANN_MLP model = CvStatModel::load( filename_to_load );
    if( model.empty() )
        cout << "Could not read the classifier " << filename_to_load << endl;
    else
        cout << "The classifier " << filename_to_load << " is loaded.\n";
    
    return model;
}
*/














