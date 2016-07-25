//
//  learn.hpp
//  iMagine
//
//  Created by Corey on 15 07, 2016.
//  Copyright © 2016 Corey. All rights reserved.
//

#ifndef learn_hpp
#define learn_hpp

#include <stdio.h>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;

class learn {

public:
    static void train();
    static void createClassifier(const string& data_filename, const string& from_file, const string& to_file);
    static void load_classifier(const string& filename_to_load);
    static vector<Mat>  getData(Mat points);
    static Mat  labelData(Mat src);
    static CvANN_MLP create_mlp(Mat trainingData, Mat trainingClasses, Mat testData, Mat testClasses);
    static Mat prepareResponses(int classSize);
//    static vector<vector<Mat>> prepareData();
      static void prepareData();
    
private:
    
    
};



#endif /* learn_hpp */
