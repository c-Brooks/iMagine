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
    Mat trainData;
    Mat trainResp;
    Mat testData;
    Mat testResp;
    int sampleSize = 946;
    
    
public:
    void   prepareData();
    Mat    getTrainData();
    Mat    getTrainResp();
    Mat    getTestData();
    Mat    getTestResp();
    void   splitData(int fulcrum);
    
    
};

#endif /* data_hpp */
