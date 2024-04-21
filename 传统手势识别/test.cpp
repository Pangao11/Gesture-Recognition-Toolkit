//
//  main.cpp
//  test
//
//  Created by yimimg pan on 2023/5/23.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include<vector>
#include<algorithm>
#include<math.h>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;
using namespace cv::ml;

Mat preprocessImage(const Mat &inputImage)
{
    Mat resizedImage, grayscaleImage;
    resize(inputImage, resizedImage, Size(128, 128));
    cvtColor(resizedImage, grayscaleImage, COLOR_BGR2GRAY);

    // 去噪
    fastNlMeansDenoising(grayscaleImage, grayscaleImage);

    // 平滑
    GaussianBlur(grayscaleImage, grayscaleImage, Size(5, 5), 0, 0);

    return grayscaleImage;
}


pair<vector<Mat>, vector<string>> readImagesFromFolder(const string& folderPath, const vector<string>& extensions)
{
    vector<Mat> images;
    vector<string> imagePaths;

    for (const auto& ext : extensions)
    {
        vector<String> fileNames;
        glob(folderPath + "/*" + ext, fileNames, false);
        for (const auto& fileName : fileNames)
        {
            Mat img = imread(fileName, IMREAD_COLOR);
            if (!img.empty())
            {
                images.push_back(preprocessImage(img));
                imagePaths.push_back(fileName);
            }
            else
            {
                cerr << "Error loading image: " << fileName << endl;
            }
        }
    }
    return make_pair(images, imagePaths);
}



// 计算HOG特征
vector<float> computeHOGFeatures(const Mat& image)
{
    // 参数设置
    Size winSize(128, 128);
    Size blockSize(8 * 16, 8 * 16);  // 3倍单元格大小
    Size blockStride(16, 16);        // 与单元格大小相同
    Size cellSize(8, 8);
    int numBins = 9;

    HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, numBins);

    vector<float> descriptors;
    hog.compute(image, descriptors);

    return descriptors;
}

int main()
{
        // 加载训练好的SVM分类器
        string svmFilename = "/Users/yimimgpan/Desktop/svm_classifier.yml";
        Ptr<SVM> svm = SVM::load(svmFilename);
        if (svm.empty())
        {
            cerr << "Failed to load SVM classifier from " << svmFilename << endl;
            return -1;
        }
        
        // 读入测试图像
        vector<string> categories = {"A", "C", "Five", "V"};
        string basePath_Test = "/Users/yimimgpan/Desktop/1/";
        vector<string> imageExtensions = {".png", ".jpg", ".jpeg", ".ppm"};
        auto [images, imagePaths] = readImagesFromFolder(basePath_Test, imageExtensions);

        glob(basePath_Test + "/*.png", imagePaths);
        
        
        // 对测试集进行预测并计算准确率
        vector<int> predictedLabels;
        for (int i = 0; i < imagePaths.size(); ++i)
        {
            Mat image = imread(imagePaths[i], IMREAD_COLOR);
            Mat preprocessedImage = preprocessImage(image);
            vector<float> hogFeatures = computeHOGFeatures(preprocessedImage);

            Mat features(1, hogFeatures.size(), CV_32F);
            for (size_t j = 0; j < hogFeatures.size(); ++j)
            {
                features.at<float>(0, j) = hogFeatures[j];
            }
            float prediction = svm->predict(features);
            predictedLabels.push_back(static_cast<int>(prediction));
            // 在图像上显示预测结果
            string predictedLabelName = categories[prediction];
            putText(image, predictedLabelName, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

            // 显示带有预测结果的图像
            string windowTitle = "Prediction: " + predictedLabelName;
            imshow(windowTitle, image);
            waitKey(300);
            destroyWindow(windowTitle);
            cout << "Predicted Label: " << predictedLabelName  << endl;
        }
        
        return 0;
}
