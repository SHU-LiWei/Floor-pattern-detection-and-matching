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

	//【1】载入原始图片
	Mat srcImage1 = imread("C:\\Users\\Administrator\\Desktop\\李伟（勿删）\\8.6\\花纹检测\\数据库图片\\2.png");
	Mat srcImage2 = imread("C:\\Users\\Administrator\\Desktop\\李伟（勿删）\\8.6\\花纹检测\\测试图片\\1.png");
	
	if( !srcImage1.data || !srcImage2.data )
	{ printf("读取图片错误，请确定目录下是否有imread函数指定的图片存在~！ \n"); return false; }  

	//【2】使用SURF算子检测关键点
	int minHessian = 400;//SURF算法中的hessian阈值
	SurfFeatureDetector detector( minHessian );//定义一个SurfFeatureDetector（SURF） 特征检测类对象  
	vector<KeyPoint> keypoints_object, keypoints_scene;//vector模板类，存放任意类型的动态数组

	//【3】调用detect函数检测出SURF特征关键点，保存在vector容器中
	detector.detect( srcImage1, keypoints_object );
	detector.detect( srcImage2, keypoints_scene );

	//【4】计算描述符（特征向量）
	SurfDescriptorExtractor extractor;
	Mat descriptors_object, descriptors_scene;
	extractor.compute( srcImage1, keypoints_object, descriptors_object );
	extractor.compute( srcImage2, keypoints_scene, descriptors_scene );

	//【5】使用FLANN匹配算子进行匹配
	FlannBasedMatcher matcher;
	vector< DMatch > matches;
	matcher.match( descriptors_object, descriptors_scene, matches );
	double max_dist = 0; double min_dist = 100;//最小距离和最大距离

	//【6】计算出关键点之间距离的最大值和最小值
	for( int i = 0; i < descriptors_object.rows; i++ )
	{ 
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

	printf(">Max dist 最大距离 : %f \n", max_dist );
	printf(">Min dist 最小距离 : %f \n", min_dist );

	//【7】存下匹配距离小于3*min_dist的点对
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
	//绘制出匹配到的关键点
	Mat img_matches;
	drawMatches( srcImage1, keypoints_object, srcImage2, keypoints_scene,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	//显示最终结果
	imshow( "效果图", img_matches );

	waitKey(0);
	return 0;
}
