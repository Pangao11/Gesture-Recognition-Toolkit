//
//  main.cpp
//  gesture recognition
//
//  Created by yimimg pan on 2023/4/10.
//

#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <ctime>
#include <algorithm>
#include <random>
#include<math.h>
#include"Header.h"
#include <filesystem>

namespace fs = std::filesystem;
using namespace std;
using namespace cv;
using namespace cv::ml;

time_t start, endP;

// 读入图像
vector<Mat> readImagesFromFolder(const string &folderPath)
{
    vector<string> extensions = {".jpg", ".jpeg", ".png", ".ppm"};
    vector<Mat> images;
    for (const auto &ext : extensions)
    {
        vector<String> fileNames;
        glob(folderPath + "/*" + ext, fileNames, false);
        for (const auto &fileName : fileNames)
        {
            Mat img = imread(fileName, IMREAD_COLOR);
            if (!img.empty())
            {
                images.push_back(preprocessImage(img));
            }
            else
            {
                cerr << "Error loading image: " << fileName << endl;
            }
        }
    }
    return images;
}

// 计算HOG特征
vector<float> computeHOGFeatures(const Mat &image)
{
    // 参数设置
    Size winSize(128, 128);
    Size blockSize(8 * 16, 8 * 16); // 3倍单元格大小
    Size blockStride(16, 16);       // 与单元格大小相同
    Size cellSize(8, 8);
    int numBins = 9;

    HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, numBins);

    vector<float> descriptors;
    hog.compute(image, descriptors);

    return descriptors;
}

int main()
{
    vector<string> categories = {"A", "C", "Five", "V"};
    string basePath_Train = "/Users/yimimgpan/Desktop/Marcel-Train/";
    string basePath_Test ="/Users/yimimgpan/Desktop/Hand_Posture_Easy_Stu/";
    vector<vector<Mat>> dataset_Train;
    vector<vector<Mat>> dataset_Test;
    
    vector<int> labels;
    
    // 初始化训练集和测试集
    vector<vector<Mat>> trainDataset(categories.size());
    vector<vector<Mat>> testDataset(categories.size());
    vector<int> trainLabels;
    vector<int> testLabels;
    
    float trainPercentage = 0.8;
    
    clock_t start, endP;
    
    start = clock();
    
    int label = 0;
    
    
    //为训练集标签
    for (const auto &category : categories)
    {
        string folderPath = basePath_Train + category;
        vector<Mat> images = readImagesFromFolder(folderPath);
        dataset_Train.push_back(images);
        
        // 为每个类别的图像分配标签
        for (size_t i = 0; i < images.size(); ++i)
        {
            labels.push_back(label);
        }
        cout << "Loaded " << images.size() << " images from category " << category << " with label " << label << endl;
        
        // 更新标签
        label++;
    }
    //为测试集标签
    label = 0;
    for (const auto &category : categories)
    {
        string folderPath = basePath_Test + category;
        vector<Mat> images = readImagesFromFolder(folderPath);
        dataset_Test.push_back(images);
        
        // 为每个类别的图像分配标签
        for (size_t i = 0; i < images.size(); ++i)
        {
            labels.push_back(label);
        }
        cout << "Loaded " << images.size() << " images from category " << category << " with label " << label << endl;
        
        // 更新标签
        label++;
    }
    
    endP = clock();
    cout << "Time: " << (double)(endP - start) / CLOCKS_PER_SEC << "s" << endl;
    
    // 将训练集导入
    label = 0;
    for (const auto &images : dataset_Train)
    {
        // 对类别内的图像进行随机排列
        vector<int> indices(images.size());
        for (auto i = 0; i < images.size(); ++i)
        {
                trainDataset[label].push_back(images[indices[i]]);
                trainLabels.push_back(label);
        }
        label++;
    }
    
    // 将测试集导入
    label = 0;
    for (const auto &images : dataset_Test)
    {
        // 对类别内的图像进行随机排列
        vector<int> indices(images.size());
        for (auto i = 0; i < images.size(); ++i)
        {
                testDataset[label].push_back(images[indices[i]]);
                testLabels.push_back(label);
        }
        label++;
    }
    
    // 计算总训练图像数量
    int numTrainImages = 0;
    for (const auto &images : trainDataset)
    {
        numTrainImages += images.size();
    }
    
    // 提取第一张图像的 HOG 特征以获取特征向量长度
    vector<float> hogFeatureExample = computeHOGFeatures(dataset_Train[0][0]);
    int featureLength = hogFeatureExample.size();
    
    // 初始化 featuresTrain 矩阵以存储所有训练图像的 HOG 特征
    Mat featuresTrain(numTrainImages, featureLength, CV_32F);
    
    // 提取所有训练图像的 HOG 特征
    int imageIndex = 0; // 训练图像索引
    for (const auto &images : trainDataset)
    {
        for (const auto &image : images)
        {
            vector<float> hogFeatures = computeHOGFeatures(image);
            for (size_t i = 0; i < hogFeatures.size(); ++i)
            {
                featuresTrain.at<float>(imageIndex, i) = hogFeatures[i];
            }
            imageIndex++;
        }
    }
    
    // 使用 featuresTrain 和 labels 训练分类器并进行预测
    // 将 labels 转换为 cv::Mat
    Mat labelsMat(numTrainImages, 1, CV_32S);
    for (size_t i = 0; i < trainLabels.size(); ++i)
    {
        labelsMat.at<int>(i, 0) = trainLabels[i];
    }
    /*
    // 创建随机森林分类器并设置参数
    Ptr<ml::RTrees> randomForest = ml::RTrees::create();
    randomForest->setMaxDepth(10);
    randomForest->setMinSampleCount(5);
    randomForest->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 0));
    randomForest->setCalculateVarImportance(true);
    randomForest->setActiveVarCount(10);
    
    // 训练随机森林分类器
    randomForest->train(featuresTrain, ml::ROW_SAMPLE, labelsMat);
    
    string randomForestFilename = "random_forest.xml";
    randomForest->save(randomForestFilename);
    cout << "Random forest saved to " << randomForestFilename << endl;
    cout << "Current working directory: " << filesystem::current_path() << endl;
     
     // 对训练集进行预测并计算准确率
     int correctPredictions = 0;
     for (int i = 0; i < featuresTrain.rows; ++i)
     {
         Mat currentFeature = featuresTrain.row(i);
         float prediction = randomForest->predict(currentFeature);
         if (prediction == trainLabels[i])
         {
             correctPredictions++;
         }
     }
     double accuracy = static_cast<double>(correctPredictions) / featuresTrain.rows;
     cout << "Training set accuracy: " << accuracy * 100 << "%" << endl;
     
     // 计算测试集的 HOG 特征
     int numTestImages = testLabels.size();
     Mat featuresTest(numTestImages, featureLength, CV_32F);
     int testImageIndex = 0;
     for (const auto &images : testDataset)
     {
         for (const auto &image : images)
         {
             vector<float> hogFeatures = computeHOGFeatures(image);
             for (size_t i = 0; i < hogFeatures.size(); i++)
             {
                 featuresTest.at<float>(testImageIndex, i) = hogFeatures[i];
             }
             testImageIndex++; }
     }
     
     // 对测试集进行预测并计算准确率
     int correctTestPredictions = 0;
     for (int i = 0; i < featuresTest.rows; ++i)
     {
         Mat currentFeature = featuresTest.row(i);
         float prediction = randomForest->predict(currentFeature);
         if (prediction == testLabels[i])
         {
             correctTestPredictions++;
         }
     }
     double testAccuracy = static_cast<double>(correctTestPredictions) / featuresTest.rows;
     cout << "Test set accuracy: " << testAccuracy * 100 << "%" << endl;
     
    */
    // 创建 SVM 分类器并设置参数
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setC(1);
    svm->setKernel(SVM::RBF); // 使用径向基核函数
    svm->setTermCriteria(TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 1000, 1e-6));

    // 训练 SVM 分类器
    svm->train(featuresTrain, ROW_SAMPLE, labelsMat);

    // 保存训练好的 SVM 分类器
    string svmFilename = "svm_classifier.yml";
    svm->save(svmFilename);
    cout << "SVM classifier saved to " << svmFilename << endl;
    
    // 获取当前工作目录
    std::string currentPath = fs::current_path().string();

    // 拼接yml文件的绝对路径
    std::string absolutePath = currentPath + "/" + svmFilename;

    // 输出绝对路径
    cout << "SVM classifier saved to " << absolutePath << endl;
     // 对训练集进行预测并计算准确率
     int correctPredictions = 0;
     for (int i = 0; i < featuresTrain.rows; ++i)
     {
         Mat currentFeature = featuresTrain.row(i);
         float prediction = svm->predict(currentFeature);
         if (prediction == trainLabels[i])
         {
             correctPredictions++;
         }
     }
     double accuracy = static_cast<double>(correctPredictions) / featuresTrain.rows;
     cout << "Training set accuracy: " << accuracy * 100 << "%" << endl;
     
     // 计算测试集的 HOG 特征
     int numTestImages = testLabels.size();
     Mat featuresTest(numTestImages, featureLength, CV_32F);
     int testImageIndex = 0;
     for (const auto &images : testDataset)
     {
         for (const auto &image : images)
         {
             vector<float> hogFeatures = computeHOGFeatures(image);
             for (size_t i = 0; i < hogFeatures.size(); i++)
             {
                 featuresTest.at<float>(testImageIndex, i) = hogFeatures[i];
             }
             testImageIndex++; }
     }
     
     // 对测试集进行预测并计算准确率
     int correctTestPredictions = 0;
     for (int i = 0; i < featuresTest.rows; ++i)
     {
         Mat currentFeature = featuresTest.row(i);
         float prediction = svm->predict(currentFeature);
         if (prediction == testLabels[i])
         {
             correctTestPredictions++;
         }
         cout<<"Predicted Label:"<<prediction<<" Actual Label:"<<testLabels[i]<<endl;
     }
     double testAccuracy = static_cast<double>(correctTestPredictions) / featuresTest.rows;
     cout << "Test set accuracy: " << testAccuracy * 100 << "%" << endl;
    return 0;
}
