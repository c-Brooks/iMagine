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


// -------------------------------------------------------------------------------------------------------
// ----------------------------------------------- iMagine -----------------------------------------------
// -------------------------------------------------------------------------------------------------------


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
    

    
    Mat gray;
    int flag;
    
    while (cin >> command)
    {
        
        if(!command.compare("cam"))
        {
            cvNamedWindow ("From Webcam", CV_WINDOW_AUTOSIZE);
            CvCapture *capture = cvCreateCameraCapture(0);
            cvStartWindowThread();
            
            while (cvWaitKey(10) != 27)
            {
                image = cvQueryFrame (capture);
                edge = operators::edgeDetect(operators::gblur(image, 7));
                
                cvtColor(image, gray, CV_BGR2GRAY);
                imshow("From Webcam", edge);
                
  
                flag = cvWaitKey(30);
                if (flag == 27) {                   // esc
                    cvDestroyWindow("From Webcam");
                    cvWaitKey(5);
                    break;
                }
            }
            
        }
        
        // TODO: green light near webcam stays lit after exited out of cam (?)
        
        if (!command.compare("open"))               // open loaded image
        {
            cvStartWindowThread();
            namedWindow(argv[1], CV_WINDOW_NORMAL);
            imshow(argv[1], image);				    // open image in separate window
            if(waitKey(0)==27){					// window stays up until esc key is pressed
                destroyWindow(argv[1]);
                waitKey(10);
            }
        }
        
        else if (!command.compare("help"))          // Display help messages
            operators::help();
        
        else if (!command.compare("new"))
            cout << " Function not yet implemented" << endl;
        
        else if (!command.compare("gblur"))         // Gaussian blur
        {
            int kern_lth = 16;
            cvStartWindowThread();

            Mat gaussIm = operators::gblur(image, kern_lth);
            
            namedWindow("Original", CV_WINDOW_NORMAL);
            imshow("Original", image);
            
            namedWindow("Blurred", CV_WINDOW_NORMAL);
            imshow("Blurred", gaussIm);				 // open image in separate window
            
            if(waitKey(0)==27){					// window stays up until esc key is pressed
            destroyWindow("Original");
            destroyWindow("Blurred");
            waitKey(10ß)
            }
        }
        
        else if (!command.compare("edge"))           // Canny edge detection
        {
            Mat edgeIm = operators::edgeDetect(operators::gblur(image, 7));
            
            namedWindow("Original", CV_WINDOW_NORMAL);
            imshow("Original", image);
            
            namedWindow("Edge", CV_WINDOW_NORMAL);
            imshow("Edge", edgeIm);
            
            if(waitKey(0)==27){					// window stays up until esc key is pressed
                destroyWindow("Original");
                destroyWindow("Edge");
                waitKey(10);
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