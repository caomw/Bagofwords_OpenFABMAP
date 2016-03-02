#ifndef BOW_FUNCTION_H
#define BOW_FUNCTION_H

#endif // BOW_FUNCTION_H


#ifndef USEDFUNCTION_H
#define USEDFUNCTION_H

#endif // USEDFUNCTION_H

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
#include <thread>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>


using namespace std;
using namespace cv;

const string paramsFile = "params.xml";
const string vocabularyFile = "vocabulary.xml.gz";
const string bowImageDescriptorsDir = "/bowImageDescriptors";




static void help(char** argv)
{
    cout << "\nThis program shows how to read in, train on and produce test results for the given dataset. \n"
     << "It shows how to use detectors, descriptors and recognition methods \n"
        "Using OpenCV version %s\n" << CV_VERSION << "\n"
     << "Call: \n"
    << "Format:\n ./" << argv[0] << " [Dataset path] [result directory]  \n"
    << "       or:  \n"
    << " ./" << argv[0] << " [Dataset path] [result directory] [feature detector] [descriptor extractor] [descriptor matcher] \n"
    << "\n"
    << "Input parameters: \n"
    << "[Dataset path]           Path to the given dataset (e.g. /home/my/VOCdevkit/VOC2010).  \n"
    << "[result directory]       Path to result diractory. Following folders will be created in [result directory]: \n"
    << "                         bowImageDescriptors - to store image descriptors, \n"
    << "                         svms - to store trained svms, \n"
    << "                         plots - to store files for plots creating. \n"
    << "[feature detector]       Feature detector name (e.g. SURF, FAST...) - see createFeatureDetector() function in detectors.cpp \n"
    << "                         Currently 2/2016, this is FAST, STAR, SIFT, SURF, MSER, GFTT, HARRIS \n"
    << "[descriptor extractor]   Descriptor extractor name (e.g. SURF, SIFT) - see createDescriptorExtractor() function in descriptors.cpp \n"
    << "                         Currently 2/2016, this is SURF, OpponentSIFT, SIFT, OpponentSURF, BRIEF \n"
    << "[descriptor matcher]     Descriptor matcher name (e.g. BruteForce) - see createDescriptorMatcher() function in matchers.cpp \n"
    << "                         Currently 2/2016, this is BruteForce, BruteForce-L1, FlannBased, BruteForce-Hamming, BruteForce-HammingLUT \n"
    << "\n";
}

static void makeDir( const string& dir )
{
#if defined WIN32 || defined _WIN32
    CreateDirectory( dir.c_str(), 0 );
#else
    mkdir( dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH );
#endif
}

static void makeUsedDirs( const string& rootPath )
{
    makeDir(rootPath + bowImageDescriptorsDir);

}

struct DDMParams
{
    DDMParams() : detectorType("SURF"), descriptorType("SURF"), matcherType("BruteForce") {}
    DDMParams( const string _detectorType, const string _descriptorType, const string& _matcherType ) :
        detectorType(_detectorType), descriptorType(_descriptorType), matcherType(_matcherType){}
    void read( const FileNode& fn )
    {
        fn["detectorType"] >> detectorType;
        fn["descriptorType"] >> descriptorType;
        fn["matcherType"] >> matcherType;
    }
    void write( FileStorage& fs ) const
    {
        fs << "detectorType" << detectorType;
        fs << "descriptorType" << descriptorType;
        fs << "matcherType" << matcherType;
    }
    void print() const
    {
        cout << "detectorType: " << detectorType << endl;
        cout << "descriptorType: " << descriptorType << endl;
        cout << "matcherType: " << matcherType << endl;
    }

    string detectorType;
    string descriptorType;
    string matcherType;
};

class ObdImage
{
public:
    ObdImage(string p_id, string p_path) : id(p_id), path(p_path) {}
    string id;
    string path;
};

static vector<ObdImage> List(const string path)
{
 boost::filesystem::path targetdir(path);
 boost::filesystem::recursive_directory_iterator iter(targetdir), end;
 string ext=".jpeg";
 vector<ObdImage> images;
 BOOST_FOREACH(boost::filesystem::path  const& i, make_pair(iter,end))
 {
     if (boost::filesystem::is_regular_file(i)&&i.extension()==ext)
     {
         ObdImage imagePath(i.filename().string(),i.string());
         images.push_back(imagePath);
     }
     //std::sort(images.begin(), images.end());
 }

 return images;
}

static vector<string> List2(const string path){
    boost::filesystem::path targetdir(path);
    boost::filesystem::recursive_directory_iterator iter(targetdir), end;
    string ext=".jpeg";
    vector<string> images;
    BOOST_FOREACH(boost::filesystem::path  const& i, make_pair(iter,end))
    {
        if (boost::filesystem::is_regular_file(i)&&i.extension()==ext)
        {
            //ObdImage imagePath(i.filename().string(),i.string());
            images.push_back(i.string());
        }
        sort(images.begin(), images.end());
    }

    return images;
}

static string getDataName(const string& dataPath)
{
    size_t found=dataPath.rfind('/');
    if( found == string::npos )
    {
        found = dataPath.rfind( '\\' );
        if( found == string::npos )
            return dataPath;
    }
    return dataPath.substr(found + 1, dataPath.size() - found);
}

static void readUsedParams(const FileNode& fn,string& dataName,DDMParams& ddmParams)
{
    fn["dataName"]>>dataName;
    FileNode currFn=fn;
    currFn=fn["ddmParams"];
    ddmParams.read(currFn);
}

static void writeUsedParams(FileStorage& fs, const string& dataName,DDMParams& ddmParams)
{
    fs<<"dataName"<<dataName;
    fs<<"ddmParams"<<"{";
    ddmParams.write(fs);
    fs<<"}";
}

static void printUsedParams( const string& vocPath, const string& resDir,
                      const DDMParams& ddmParams )
{
    cout << "CURRENT CONFIGURATION" << endl;
    cout << "----------------------------------------------------------------" << endl;
    cout << "vocPath: " << vocPath << endl;
    cout << "resDir: " << resDir << endl;
    cout << endl; ddmParams.print();
    cout << "----------------------------------------------------------------" << endl << endl;
}




static bool readVocabulary( const string& filename, Mat& vocabulary )
{
    cout << "Reading vocabulary...";
    FileStorage fs( filename, FileStorage::READ );
    if( fs.isOpened() )
    {
        fs["vocabulary"] >> vocabulary;
        cout << "done" << endl;
        return true;
    }
    return false;
}

static bool writeVocabulary( const string& filename, const Mat& vocabulary )
{
    cout << "Saving vocabulary..." << endl;
    FileStorage fs( filename, FileStorage::WRITE );
    if( fs.isOpened() )
    {
        fs << "vocabulary" << vocabulary;
        return true;
    }
    return false;
}


static Mat trainVocabulary(const string& filename, const Ptr<FeatureDetector>& fdetector, const Ptr<DescriptorExtractor>& dextractor )
{
    Mat vocabulary;
    if(!readVocabulary(filename,vocabulary))
    {
        cout<<"Computing descriptors..."<<endl;
    RNG& rng=theRNG();
    TermCriteria terminate_criterion;
    terminate_criterion.epsilon=FLT_EPSILON;
    BOWKMeansTrainer bowTrainer(545, terminate_criterion,3,KMEANS_PP_CENTERS);
    vector<ObdImage> images=List("/home/jiaqi/Pictures/fabmap");
    while(images.size()>0)
    {
        int randImgIdx=rng((unsigned)images.size());
        Mat colorImage=imread(images[randImgIdx].path);
        vector<KeyPoint> imageKeypoints;
        fdetector->detect(colorImage,imageKeypoints);
        Mat imageDescriptors;
        dextractor->compute(colorImage,imageKeypoints,imageDescriptors);
        bowTrainer.add(imageDescriptors);
        cout<<images.size()<<"images left, "<<images[randImgIdx].id<<" processed."<<endl;
        images.erase( images.begin() + randImgIdx );
    }
    cout<<"Training vocabulary..."<<endl;
    vocabulary=bowTrainer.cluster();
    if( !writeVocabulary(filename, vocabulary) )
    {
        cout << "Error: file " << filename << " can not be opened to write" << endl;
        exit(-1);
    }
}
    return vocabulary;
}




// Load in the bag of words vectors for a set of images, from file if possible
//static void calculateImageDescriptors( const vector<ObdImage>& images, vector<Mat>& imageDescriptors,
//                                Ptr<BOWImgDescriptorExtractor>& bowExtractor, const Ptr<FeatureDetector>& fdetector,
//                                const string& resPath )
//{  // Ptr<DescriptorMatcher> testMatcher=DescriptorMatcher::create("BruteForce");

//    CV_Assert( !bowExtractor->getVocabulary().empty() );
//    imageDescriptors.resize( images.size() );

//    for( size_t i = 0; i < images.size(); i++ )
//    {
//        string filename = resPath + bowImageDescriptorsDir + "/" + images[i].id + ".xml.gz";
//        if( readBowImageDescriptor( filename, imageDescriptors[i] ) )
//        {
////#ifdef DEBUG_DESC_PROGRESS
////            cout << "Loaded bag of word vector for image " << i+1 << " of " << images.size() << " (" << images[i].id << ")" << endl;
////#endif
//        }
//        else
//        {
//            Mat colorImage = imread( images[i].path );
////#ifdef DEBUG_DESC_PROGRESS
////            cout << "Computing descriptors for image " << i+1 << " of " << images.size() << " (" << images[i].id << ")" << flush;
////#endif
//            vector<KeyPoint> keypoints;
//            fdetector->detect( colorImage, keypoints );
////            Mat originalDescriptors;
////            dextractor->compute(colorImage, keypoints,originalDescriptors);
////#ifdef DEBUG_DESC_PROGRESS
////                cout << " + generating BoW vector" << std::flush;
////#endif
//            bowExtractor->compute( colorImage, keypoints, imageDescriptors[i] );
////            vector<DMatch> matches;
////            testMatcher->match(originalDescriptors,imageDescriptors[i],matches);
////#ifdef DEBUG_DESC_PROGRESS
////            cout << " ...DONE " << static_cast<int>(static_cast<float>(i+1)/static_cast<float>(images.size())*100.0)
////                 << " % complete" << endl;
////#endif
//            if( !imageDescriptors[i].empty() )
//            {
//                if( !writeBowImageDescriptor( filename, imageDescriptors[i] ) )
//                {
//                    cout << "Error: file " << filename << "can not be opened to write bow image descriptor" << endl;
//                    exit(-1);
//                }
//            }
//        }
//    }
//}

static void calculateImageDescriptors( const vector<string>& images, Mat& imageDescriptors,
                                Ptr<BOWImgDescriptorExtractor>& bowExtractor, const Ptr<FeatureDetector>& fdetector)
{



    for( size_t i = 0; i < images.size(); i++ )
    {


            Mat colorImage = imread( images[i] );

            vector<KeyPoint> keypoints;
            fdetector->detect( colorImage, keypoints );
            Mat bow;
            bowExtractor->compute( colorImage, keypoints, bow );
            imageDescriptors.push_back(bow);
            drawKeypoints(colorImage, keypoints, colorImage);
            imshow(images[i], colorImage);
            waitKey(10);

    }
}
