//
//  data.hpp
//  iMagine
//
//  Created by Corey on 25 07, 2016.
//  Copyright © 2016 Corey. All rights reserved.
//

#ifndef data_hpp
#define data_hpp

#include <stdio.h>
#include <stdio.h>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;

class data {
    
private:
    vector<Mat> trainData;
    vector<Mat> trainResp;
    
public:
    void   prepareData();
    vector<Mat>    getTrainData();
    vector<Mat>    getTrainResp();
    Mat            dataAt(int i);
    Mat            respAt(int i);
};

#endif /* data_hpp */
