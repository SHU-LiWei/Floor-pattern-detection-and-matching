#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <iostream>
using namespace cv;
using namespace std;

int main(  )
{

	//��1������ԭʼͼƬ
	Mat srcImage1 = imread("C:\\Users\\Administrator\\Desktop\\��ΰ����ɾ��\\8.6\\���Ƽ��\\���ݿ�ͼƬ\\2.png");
	Mat srcImage2 = imread("C:\\Users\\Administrator\\Desktop\\��ΰ����ɾ��\\8.6\\���Ƽ��\\����ͼƬ\\1.png");
	
	if( !srcImage1.data || !srcImage2.data )
	{ printf("��ȡͼƬ������ȷ��Ŀ¼���Ƿ���imread����ָ����ͼƬ����~�� \n"); return false; }  

	//��2��ʹ��SURF���Ӽ��ؼ���
	int minHessian = 400;//SURF�㷨�е�hessian��ֵ
	SurfFeatureDetector detector( minHessian );//����һ��SurfFeatureDetector��SURF�� ������������  
	vector<KeyPoint> keypoints_object, keypoints_scene;//vectorģ���࣬����������͵Ķ�̬����

	//��3������detect��������SURF�����ؼ��㣬������vector������
	detector.detect( srcImage1, keypoints_object );
	detector.detect( srcImage2, keypoints_scene );

	//��4������������������������
	SurfDescriptorExtractor extractor;
	Mat descriptors_object, descriptors_scene;
	extractor.compute( srcImage1, keypoints_object, descriptors_object );
	extractor.compute( srcImage2, keypoints_scene, descriptors_scene );

	//��5��ʹ��FLANNƥ�����ӽ���ƥ��
	FlannBasedMatcher matcher;
	vector< DMatch > matches;
	matcher.match( descriptors_object, descriptors_scene, matches );
	double max_dist = 0; double min_dist = 100;//��С�����������

	//��6��������ؼ���֮���������ֵ����Сֵ
	for( int i = 0; i < descriptors_object.rows; i++ )
	{ 
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

	printf(">Max dist ������ : %f \n", max_dist );
	printf(">Min dist ��С���� : %f \n", min_dist );

	//��7������ƥ�����С��3*min_dist�ĵ��
	std::vector< DMatch > good_matches;
	for( int i = 0; i < descriptors_object.rows; i++ )
	{ 
		if( matches[i].distance < 3*min_dist )
		{ 
			good_matches.push_back( matches[i]);
		}
	}
	cout << good_matches.size() << endl;
	double sum = 0;
	//sort(good_matches.begin(), good_matches.end());
	for (int i = 0; i < 78; ++i)
	{
		sum += good_matches[i].distance;
	}
	cout << sum << endl;
	//���Ƴ�ƥ�䵽�Ĺؼ���
	Mat img_matches;
	drawMatches( srcImage1, keypoints_object, srcImage2, keypoints_scene,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	//��ʾ���ս��
	imshow( "Ч��ͼ", img_matches );

	waitKey(0);
	return 0;
}
