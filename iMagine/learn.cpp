//
//  learn.cpp
//  iMagine
//
//  Created by Corey on 15 07, 2016.
//  Copyright © 2016 Corey. All rights reserved.
//

#include "learn.hpp"

#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <opencv/ml.h>

#include <cstdio>
#include <vector>
#include <iostream>


using namespace cv;
using namespace std;



Mat learn::getData(Mat points){
    namedWindow("TEST", CV_WINDOW_AUTOSIZE);
    Mat image = imread("3.jpg", CV_32FC1);
    threshold(image, image, 100, 255, 0);
    resize(image, image, Size(points.rows, points.rows));
    
    // I don't know why, but I get a segmentation error 11
    //      if I don't open the image immediately.
    imshow("TEST", image);
    destroyWindow("TEST");
    waitKey(10);
    
    return image;
}



Mat learn::labelData(Mat points){
    Mat labels(points.rows, 1, CV_32FC1);
    Mat image = learn::getData(points);
    
    for(int i = 0; i < points.rows; i++) {
        float x = points.at<float>(i,0) * points.rows;
        float y = points.at<float>(i,1) * points.rows;
        labels.at<float>(i, 0) = image.at<int>(Point(x,y)) >= 0 ? -1 : 1;
        cout << Point(x,y) << endl;
        cout << image.at<int>(Point(x,y)) << endl;
        
    }
    return labels;
}





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














