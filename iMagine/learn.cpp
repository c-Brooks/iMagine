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
    }
    return labels;
}





Mat learn::mlp(Mat trainingData, Mat trainingClasses, Mat testData, Mat testClasses) {
    
    Mat layers = cv::Mat(4, 1, CV_32SC1);
    
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
    mlp.train(trainingData, trainingClasses, Mat(), Mat(), params);
    
    cv::Mat response(1, 1, CV_32FC1);
    cv::Mat predicted(testClasses.rows, 1, CV_32F);
    for(int i = 0; i < testData.rows; i++) {
        cv::Mat response(1, 1, CV_32FC1);
        cv::Mat sample = testData.row(i);
        
        mlp.predict(sample, response);
        predicted.at<float>(i,0) = response.at<float>(0,0);
    }
    mlp.save("");
    return predicted;
}

