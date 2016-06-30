//
//  main.cpp
//  iMagine
//
//  Created by Corey Brooks on 2016-06-29.
//  Copyright © 2016 Corey Brooks. All rights reserved.

#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <iostream>

#include "operators.hpp"

using namespace cv;
using namespace std;


// ------------------------------------------------------------------------------------------------------- //

int main(int argc, char** argv)
{
    
    Mat image, edge;
    String command;
    
    
    if (argc != 2)
    {
        cerr << "\n\n Error: no argument specified" << endl;
        return -1;
    }
    
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    
    if (!image.data)
    {
        cerr << "\n  Could not open or find the image" << endl;
        return -1;
    }
    else cout << " Succesfully loaded " << argv[1] << endl;
    
    cout << "\n ----------------------------------------" << endl
         <<   " --------- Welcome to iMagine -----------" << endl
         <<   " ----------------------------------------" << endl;
    
    
    cout << " For help, type 'help' \n" << endl;
    
    cvNamedWindow ("Capture", CV_WINDOW_AUTOSIZE);
    CvCapture *capture = cvCreateCameraCapture(0);

    // Input from webcam is traced and output to screen
    while (1)
    {
    image = cvQueryFrame (capture);
    edge = operators::edgeDetect(operators::gblur(image, 5));
    imshow("Capture", edge);
    }
    
    
    while (cin >> command)
    {
        
        if (!command.compare("open"))
        {
            namedWindow(argv[1], CV_WINDOW_NORMAL);
            imshow(argv[1], edge);				    // open image in separate window
            waitKey(0);								// window stays up until key is pressed
            destroyWindow(argv[1]);
        }
        
        else if (!command.compare("help"))
            operators::help(); // Display help messages
        
        else if (!command.compare("new"))
            cout << " Function not yet implemented" << endl;
        
        else if (!command.compare("gblur")) // Gaussian blur
        {
            int kern_lth = 16;
            
            Mat gaussIm = operators::gblur(image, kern_lth);
            
            namedWindow("Original", CV_WINDOW_NORMAL);
            imshow("Original", image);
            
            namedWindow("Blurred", CV_WINDOW_NORMAL);
            imshow("Blurred", gaussIm);				    // open image in separate window
            
            waitKey(0);
            destroyWindow("Original");
            destroyWindow("Blurred");
            
        }
        
        else if (!command.compare("edge")) // Canny edge detection
        {
            Mat edgeIm = operators::edgeDetect(operators::gblur(image, 8));
            
            namedWindow("Original", CV_WINDOW_NORMAL);
            imshow("Original", image);
            
            namedWindow("Edge", CV_WINDOW_NORMAL);
            imshow("Edge", edgeIm);
            
            waitKey(0);
            destroyWindow("Original");
            destroyWindow("Edge");
        }
        
        else if (!command.compare("close")){
            destroyAllWindows();
        }
        
        else if (!command.compare("bye")){
            destroyAllWindows();
            return 0;
        }
        else
            cout << " Not a valid command.\n For help, type help." << endl;
    }
    destroyAllWindows();
    return 0;
}