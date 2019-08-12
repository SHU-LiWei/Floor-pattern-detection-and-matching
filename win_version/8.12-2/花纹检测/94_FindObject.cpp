#include <iostream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include<opencv2/legacy/legacy.hpp>
#include <vector>
#include <string>
#include <ctime>
#include <fstream>
#include <stdio.h>
using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
	clock_t time_stt = clock();
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	vector<DMatch> matches;
	std::vector<DMatch> good_matches;
	int dataNumber = 10;
	//-- 读取图像

	char filename[100];
	vector<Mat> images_Database;
	for (int i = 0; i < dataNumber; i++)
	{
		sprintf_s(filename, "C:\\Users\\Administrator\\Desktop\\李伟（勿删）\\8.12-2\\花纹检测\\数据库图片\\%d.png", i);//用这个函数来转换图片名称，存放在filename中。
		images_Database.push_back(imread(filename, 1));
	}
	
	Mat img_1 = imread("C:\\Users\\Administrator\\Desktop\\李伟（勿删）\\8.12-2\\花纹检测\\测试图片\\0.png");
	
	//第一次循环寻找筛选后特征点数量最小值

	vector<int > min_GoodMatches;

	for (int j = 0; j < dataNumber; ++j) {
		Mat img_2 = images_Database[j];
	
		//-- 初始化
		//std::vector<KeyPoint> keypoints_1, keypoints_2;
		
		Mat descriptors_1, descriptors_2;
		Ptr<FeatureDetector> detector = ORB::create("ORB");
		Ptr<DescriptorExtractor> descriptor = ORB::create("ORB");
		// Ptr<FeatureDetector> detector = FeatureDetector::create(detector_name);
		// Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create(descriptor_name);
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

		//-- 第一步:检测 Oriented FAST 角点位置
		detector->detect(img_1, keypoints_1);
		detector->detect(img_2, keypoints_2);

		//-- 第二步:根据角点位置计算 BRIEF 描述子
		descriptor->compute(img_1, keypoints_1, descriptors_1);
		descriptor->compute(img_2, keypoints_2, descriptors_2);

		//-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
		//vector<DMatch> matches;
		//BFMatcher matcher ( NORM_HAMMING );
		matcher->match(descriptors_1, descriptors_2, matches);

		//-- 第四步:匹配点对筛选
		double min_dist = 10000;

		//找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
		for (int i = 0; i < descriptors_1.rows; i++) {
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
		}

		// 求最小距离
		min_dist = min_element(matches.begin(), matches.end(),
			[](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; })->distance;

		//当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值50作为下限.
		//std::vector<DMatch> good_matches;
		for (int i = 0; i < descriptors_1.rows; i++) {

			// 注意：要与第二个循环保持一致

			if (matches[i].distance <= max(3 * min_dist, 60.0)) {
				good_matches.push_back(matches[i]);
			}
		}

		min_GoodMatches.push_back(good_matches.size());
		
		keypoints_1.clear();
		keypoints_2.clear();
		matches.clear();
		good_matches.clear();
	}

	int min_GoodMatches_size = *min_element(min_GoodMatches.begin(), min_GoodMatches.end());
	cout << "最小匹配点数" << min_GoodMatches_size << endl;
	vector<double > score;
	for (int j = 0; j < dataNumber; ++j) {

		Mat img_2 = images_Database[j];
		//-- 初始化
		//std::vector<KeyPoint> keypoints_1, keypoints_2;
		Mat descriptors_1, descriptors_2;
		Ptr<FeatureDetector> detector = ORB::create("ORB");
		Ptr<DescriptorExtractor> descriptor = ORB::create("ORB");
		// Ptr<FeatureDetector> detector = FeatureDetector::create(detector_name);
		// Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create(descriptor_name);
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

		//-- 第一步:检测 Oriented FAST 角点位置
		detector->detect(img_1, keypoints_1);
		detector->detect(img_2, keypoints_2);

		//-- 第二步:根据角点位置计算 BRIEF 描述子
		descriptor->compute(img_1, keypoints_1, descriptors_1);
		descriptor->compute(img_2, keypoints_2, descriptors_2);

		//-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
		//vector<DMatch> matches;
		//BFMatcher matcher ( NORM_HAMMING );
		matcher->match(descriptors_1, descriptors_2, matches);

		//-- 第四步:匹配点对筛选
		double min_dist = 10000;

		//找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
		for (int i = 0; i < descriptors_1.rows; i++) {
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
		}

		// 求最小距离
		min_dist = min_element(matches.begin(), matches.end(),
			[](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; })->distance;

		cout << "第" << (j + 1) << "类" << "图片匹配最小距离" << min_dist << endl;

		//当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值50作为下限.
		//std::vector<DMatch> good_matches;
		for (int i = 0; i < descriptors_1.rows; i++) {

			// 注意：要与第１个循环保持一致

			if (matches[i].distance <= max(3 * min_dist, 60.0)) {
				good_matches.push_back(matches[i]);
			}
		}

		double sum = 0;
		cout << "第" << (j + 1) << "类" << "匹配点数：" << good_matches.size() << endl;

		//sort(good_matches.begin(), good_matches.end());

		for (int i = 0; i < 48; ++i) {
			sum += good_matches[i].distance;
		}
		cout << "第" << (j + 1) << "类" << "图片匹配得分：" << sum << endl << endl;

		score.push_back(sum);
		keypoints_1.clear();
		keypoints_2.clear();
		matches.clear();
		good_matches.clear();
	}
	int  minPosition = min_element(score.begin(), score.end()) - score.begin();

	cout << "检测结果为第" << minPosition + 1 << "类图片" << endl;

	cout << "用时" << (clock() - time_stt) / (double)CLOCKS_PER_SEC << " s" << endl;

	// write result to result.txt
	ofstream ofs;
	ofs.open("C:\\Users\\Administrator\\Desktop\\李伟（勿删）\\8.12-2\\花纹检测\\result.txt", ios::out | ios::trunc);
	if (!ofs.is_open())
	{
		cout << "存储结果失败" << endl;
	}
	ofs << (minPosition + 1) << endl;
	ofs.close();

	return 0;
}
