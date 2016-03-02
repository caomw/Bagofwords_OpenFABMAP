#ifndef FABMAP_FUNCTION_H
#define FABMAP_FUNCTION_H

#endif // FABMAP_FUNCTION_H
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/ml/ml.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <functional>
#include <sys/stat.h>
#include <stdio.h>
#include <sys/types.h>
#include <string>
#include <string.h>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

using namespace std;
using namespace cv;


void readData(const string& filename, Mat& data, const string& content){
    FileStorage fs( filename, FileStorage::READ );
    if( fs.isOpened() )
    {
        fs[content] >> data;
        cout << "done" << endl;

    }
}
