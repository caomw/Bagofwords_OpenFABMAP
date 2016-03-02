#include <QCoreApplication>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/opencv.hpp"
#include "bow_function.h"
#include "fabmap_function.h"

using namespace std;
using namespace cv;

const string vocpath ="vocab_small.yml";
const string trainpath="train_data_small.yml";


int main(int argc, char *argv[])
{
        if( argc != 3 && argc != 6 )
        {
            help(argv);
            return -1;
        }

        initModule_nonfree();
        const string dataPath=argv[1],resPath=argv[2];
        string dataName;
        DDMParams ddmParams;

        makeUsedDirs(resPath);

        FileStorage paramsFS(resPath+"/"+paramsFile,FileStorage::READ);
        if(paramsFS.isOpened()){
            readUsedParams(paramsFS.root(),dataName,ddmParams);
        }else
        {
            dataName=getDataName(dataPath);
            if(argc!=6)
            {
                cout<<"Feature detector, descriptor extractor, descriptor matcher must be set"<<endl;
                return -1;
            }
            ddmParams=DDMParams(argv[3],argv[4],argv[5]);
            paramsFS.open(resPath+"/"+paramsFile,FileStorage::WRITE);
            if(paramsFS.isOpened())
            {
                writeUsedParams(paramsFS,dataName,ddmParams);
            }
            else
            {
                cout<<"File"<<(resPath+"/"+paramsFile)<<"can not be opened to write"<<endl;
                return -1;
            }

        }
         paramsFS.release();
        //Create detector, descriptor, matcher.

        Ptr<FeatureDetector> featureDetector=FeatureDetector::create(ddmParams.detectorType);
        Ptr<DescriptorExtractor> descExtractor=DescriptorExtractor::create(ddmParams.descriptorType);
        Ptr<DescriptorMatcher> descMatcher=DescriptorMatcher::create(ddmParams.matcherType);
        Ptr<BOWImgDescriptorExtractor> bowExtractor;
        bowExtractor=new BOWImgDescriptorExtractor(descExtractor,descMatcher);

        //Print configureation to screen.
        printUsedParams(dataPath,resPath,ddmParams);
        //Train the vocabulary
        double t=getTickCount();
        Mat vocabulary=trainVocabulary(resPath+"/"+vocabularyFile,
                                       featureDetector,descExtractor);
        t=(getTickCount()-t)/getTickFrequency();
        cout<<"Times for training vocabulary in seconds: "<<t<<"s = "<<t/60<<"mins"<<endl;
        bowExtractor->setVocabulary(vocabulary);
        //Compute the bag of words description for test images
//        vector<ObdImage> test_images;
//        ObdImage t1("2010_000001.jpg","/home/jiaqi/VOCdevkit/VOC2010/JPEGImages/2010_000001.jpg");
//        ObdImage t2("2010_001263.jpg","/home/jiaqi/VOCdevkit/VOC2010/JPEGImages/2010_001263.jpg");
//        test_images.push_back(t1);
//        test_images.push_back(t2);
//        vector<Mat> imageDescriptors;
//        calculateImageDescriptors( test_images, imageDescriptors, bowExtractor, featureDetector, resPath );

        //Begin to use Fabmap
        cout<<"Now begin the FAB-MAP image matching algorithm."<<endl;
        cout<<"Loading Vocabulary: "<<endl;
        Mat vocab;
        readData(dataPath+"/"+vocpath,
                 vocab,"Vocabulary");
        cout<<"Loading Training Data: "<<endl;
        Mat trainData;
        readData(dataPath+"/"+trainpath,trainData,"BOWImageDescs");
        //Create Chow_Liu tree
        cout << "Making Chow-Liu Tree from training data" << endl <<endl;
        of2::ChowLiuTree treeBuilder;
        treeBuilder.add(trainData);
        Mat tree=treeBuilder.make();

        //Generate test data
        cout<<"Extracting Test Data from images"<<endl<<endl;
        Ptr<FeatureDetector> detector =
            new DynamicAdaptedFeatureDetector(
            AdjusterAdapter::create("STAR"), 130, 150, 5);
        Ptr<DescriptorExtractor> extractor =
            new SurfDescriptorExtractor(1000, 4, 2, false, true);
        Ptr<DescriptorMatcher> matcher =
            DescriptorMatcher::create("FlannBased");
        Ptr<BOWImgDescriptorExtractor> bide;
        bide=new BOWImgDescriptorExtractor(extractor, matcher);

        bide->setVocabulary(vocab);

        vector<string> imageNames=List2(dataPath);

        Mat testData;
        calculateImageDescriptors(imageNames,testData,bide,detector);
        //run fabmap
        cout<<"Running FAB-MAP algorithm"<<endl<<endl;
        Ptr<of2::FabMap> fabmap;
        fabmap=new of2::FabMap2(tree,0.39,0,of2::FabMap::SAMPLED | of2::FabMap::CHOW_LIU);
        fabmap->addTraining(trainData);

        vector<of2::IMatch> matches;
        fabmap->compare(testData, matches, true);

        //display output
        Mat result_small = Mat::zeros(10, 10, CV_8UC1);
        vector<of2::IMatch>::iterator l;

        for(l = matches.begin(); l != matches.end(); l++) {
                if(l->imgIdx < 0) {
                    result_small.at<char>(l->queryIdx, l->queryIdx) =
                        (char)(l->match*255);

                } else {
                    result_small.at<char>(l->queryIdx, l->imgIdx) =
                        (char)(l->match*255);
                }
        }

        Mat result_large(100, 100, CV_8UC1);
        resize(result_small, result_large, Size(500, 500), 0, 0, CV_INTER_NN);

        cout << endl << "Press any key to exit" << endl;
        imshow("Confusion Matrix", result_large);

        //visualizing the location points
        Mat  bg(500,500,CV_8U,Scalar(255));
        vector<of2::IMatch> goodMatches;
        vector<of2::IMatch>::iterator l1;
        for(l1=matches.begin();l1!=matches.end();l1++){
            if(l1->match>0.5){
                goodMatches.push_back(*l1);
            }
        }
        vector<Point> pt;
        vector<of2::IMatch>::iterator l2;
        Point ptBegin(10,250);
        circle(bg,ptBegin,5,0);
        Point ptS(ptBegin.x-10,ptBegin.y+30);
        putText(bg,"1",ptS,0,1,0);
        for(l2=goodMatches.begin()+1;l2!=goodMatches.end();l2++){

                pt.push_back(ptBegin);
                Point ptEnd(ptBegin.x+50,ptBegin.y);
                circle(bg,ptEnd,5,0);
                line(bg,ptBegin,ptEnd,0);
                Point ptS(ptEnd.x-10,ptEnd.y+30);
                stringstream ss;
                ss<<l2->queryIdx+1;
                string str=ss.str();
                putText(bg,str,ptS,0,1,0);

                ptBegin=ptEnd;

        }
        for(l2=goodMatches.begin();l2!=goodMatches.end();l2++){

                if(l2->imgIdx>0){
                    int pid=l2->imgIdx;
                    int qid=l2->queryIdx;
                    Point ptB=pt[pid];
                    Point ptE=pt[qid];
                    Point ptC((ptB.x+ptE.x)/2,ptB.y);
                    float r=(ptE.x-ptB.x)/2;
                    circle(bg,ptC,r,150);

                }

        }
        imshow("Visiualing location", bg);
        waitKey();

        return 0;

}
