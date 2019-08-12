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
	//-- ��ȡͼ��

	char filename[100];
	vector<Mat> images_Database;
	for (int i = 0; i < dataNumber; i++)
	{
		sprintf_s(filename, "C:\\Users\\Administrator\\Desktop\\��ΰ����ɾ��\\8.12-2\\���Ƽ��\\���ݿ�ͼƬ\\%d.png", i);//�����������ת��ͼƬ���ƣ������filename�С�
		images_Database.push_back(imread(filename, 1));
	}
	
	Mat img_1 = imread("C:\\Users\\Administrator\\Desktop\\��ΰ����ɾ��\\8.12-2\\���Ƽ��\\����ͼƬ\\0.png");
	
	//��һ��ѭ��Ѱ��ɸѡ��������������Сֵ

	vector<int > min_GoodMatches;

	for (int j = 0; j < dataNumber; ++j) {
		Mat img_2 = images_Database[j];
	
		//-- ��ʼ��
		//std::vector<KeyPoint> keypoints_1, keypoints_2;
		
		Mat descriptors_1, descriptors_2;
		Ptr<FeatureDetector> detector = ORB::create("ORB");
		Ptr<DescriptorExtractor> descriptor = ORB::create("ORB");
		// Ptr<FeatureDetector> detector = FeatureDetector::create(detector_name);
		// Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create(descriptor_name);
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

		//-- ��һ��:��� Oriented FAST �ǵ�λ��
		detector->detect(img_1, keypoints_1);
		detector->detect(img_2, keypoints_2);

		//-- �ڶ���:���ݽǵ�λ�ü��� BRIEF ������
		descriptor->compute(img_1, keypoints_1, descriptors_1);
		descriptor->compute(img_2, keypoints_2, descriptors_2);

		//-- ������:������ͼ���е�BRIEF�����ӽ���ƥ�䣬ʹ�� Hamming ����
		//vector<DMatch> matches;
		//BFMatcher matcher ( NORM_HAMMING );
		matcher->match(descriptors_1, descriptors_2, matches);

		//-- ���Ĳ�:ƥ����ɸѡ
		double min_dist = 10000;

		//�ҳ�����ƥ��֮�����С�����������, ���������Ƶĺ�����Ƶ������֮��ľ���
		for (int i = 0; i < descriptors_1.rows; i++) {
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
		}

		// ����С����
		min_dist = min_element(matches.begin(), matches.end(),
			[](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; })->distance;

		//��������֮��ľ��������������С����ʱ,����Ϊƥ������.����ʱ����С�����ǳ�С,����һ������ֵ50��Ϊ����.
		//std::vector<DMatch> good_matches;
		for (int i = 0; i < descriptors_1.rows; i++) {

			// ע�⣺Ҫ��ڶ���ѭ������һ��

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
	cout << "��Сƥ�����" << min_GoodMatches_size << endl;
	vector<double > score;
	for (int j = 0; j < dataNumber; ++j) {

		Mat img_2 = images_Database[j];
		//-- ��ʼ��
		//std::vector<KeyPoint> keypoints_1, keypoints_2;
		Mat descriptors_1, descriptors_2;
		Ptr<FeatureDetector> detector = ORB::create("ORB");
		Ptr<DescriptorExtractor> descriptor = ORB::create("ORB");
		// Ptr<FeatureDetector> detector = FeatureDetector::create(detector_name);
		// Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create(descriptor_name);
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

		//-- ��һ��:��� Oriented FAST �ǵ�λ��
		detector->detect(img_1, keypoints_1);
		detector->detect(img_2, keypoints_2);

		//-- �ڶ���:���ݽǵ�λ�ü��� BRIEF ������
		descriptor->compute(img_1, keypoints_1, descriptors_1);
		descriptor->compute(img_2, keypoints_2, descriptors_2);

		//-- ������:������ͼ���е�BRIEF�����ӽ���ƥ�䣬ʹ�� Hamming ����
		//vector<DMatch> matches;
		//BFMatcher matcher ( NORM_HAMMING );
		matcher->match(descriptors_1, descriptors_2, matches);

		//-- ���Ĳ�:ƥ����ɸѡ
		double min_dist = 10000;

		//�ҳ�����ƥ��֮�����С�����������, ���������Ƶĺ�����Ƶ������֮��ľ���
		for (int i = 0; i < descriptors_1.rows; i++) {
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
		}

		// ����С����
		min_dist = min_element(matches.begin(), matches.end(),
			[](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; })->distance;

		cout << "��" << (j + 1) << "��" << "ͼƬƥ����С����" << min_dist << endl;

		//��������֮��ľ��������������С����ʱ,����Ϊƥ������.����ʱ����С�����ǳ�С,����һ������ֵ50��Ϊ����.
		//std::vector<DMatch> good_matches;
		for (int i = 0; i < descriptors_1.rows; i++) {

			// ע�⣺Ҫ��ڣ���ѭ������һ��

			if (matches[i].distance <= max(3 * min_dist, 60.0)) {
				good_matches.push_back(matches[i]);
			}
		}

		double sum = 0;
		cout << "��" << (j + 1) << "��" << "ƥ�������" << good_matches.size() << endl;

		//sort(good_matches.begin(), good_matches.end());

		for (int i = 0; i < 48; ++i) {
			sum += good_matches[i].distance;
		}
		cout << "��" << (j + 1) << "��" << "ͼƬƥ��÷֣�" << sum << endl << endl;

		score.push_back(sum);
		keypoints_1.clear();
		keypoints_2.clear();
		matches.clear();
		good_matches.clear();
	}
	int  minPosition = min_element(score.begin(), score.end()) - score.begin();

	cout << "�����Ϊ��" << minPosition + 1 << "��ͼƬ" << endl;

	cout << "��ʱ" << (clock() - time_stt) / (double)CLOCKS_PER_SEC << " s" << endl;

	// write result to result.txt
	ofstream ofs;
	ofs.open("C:\\Users\\Administrator\\Desktop\\��ΰ����ɾ��\\8.12-2\\���Ƽ��\\result.txt", ios::out | ios::trunc);
	if (!ofs.is_open())
	{
		cout << "�洢���ʧ��" << endl;
	}
	ofs << (minPosition + 1) << endl;
	ofs.close();

	return 0;
}
