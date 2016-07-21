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
    static Mat  getData(Mat points);
    static Mat  labelData(Mat src);
    static Mat  mlp(Mat trainingData, Mat trainingClasses, Mat testData, Mat testClasses);
    
private:
    
    
};



#endif /* learn_hpp */
