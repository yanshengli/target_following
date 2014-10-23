// CarDetectAndTrack.cpp : 定义控制台应用程序的入口点。
//


#include <iostream>
#include <opencv2/opencv.hpp>
#include <cv.h>
#include<highgui.h>

using namespace std;
using namespace cv;

#define VideoFile "video1.avi"
#define TrainNo 50
#define ZERO 0.000001
#define PI 3.141593
#define SLOTnum 10
#define Thre1 1
#define Thre2 1
#define Thre3 1
#define Thre4 1

struct feature{
double density;
double motion_strength;
};

vector<struct feature> feature_store;

//对轮廓按面积降序排列
bool biggerSort(vector<cv::Point> v1, vector<cv::Point> v2)
{
	return cv::contourArea(v1)>cv::contourArea(v2);
}

Mat shade_detect(Mat hsv,Mat back_hsv,Mat foreground)
{
int height=foreground.rows;
int width=foreground.cols;
int no=0;
for(int i=0;i<height;i++)
	for(int j=0;j<width;j++)
	{
	if(	(hsv.at<uchar>(i,(3*j+2))-back_hsv.at<uchar>(i,(3*j+2))<0) &&
		(1<(back_hsv.at<uchar>(i,(3*j+2))/(hsv.at<uchar>(i,(3*j+2))+1)))&&
		(back_hsv.at<uchar>(i,(3*j+2))/(hsv.at<uchar>(i,(3*j+2))+1)<=3)&&
		(hsv.at<uchar>(i,(3*j+1))-back_hsv.at<uchar>(i,(3*j+1))<0.15)&&
		(abs(hsv.at<uchar>(i,(3*j+0))-back_hsv.at<uchar>(i,(3*j+0)))<=0.7)	)
		{
			foreground.at<uchar>(i,j)=0;
			no++;
		}
	}

//cout<<no;
return foreground;

}


double den_cal(Mat forground)
{
int number=countNonZero(forground);
return double(number)/double((forground.rows*forground.cols));

}


int judge(vector<struct feature> info,double density,double stren)
{

density=double(density/SLOTnum);
stren=double(stren/SLOTnum);
double accm_density=0.0;
double accm_stren=0.0;

for_each(begin(info),end(info),[&](struct feature data)
{accm_density+=(data.density-density)*(data.density-density);});


for_each(begin(info),end(info),[&](struct feature data)
{accm_stren+=(data.motion_strength-stren)*(data.motion_strength-stren);});


double result_density=sqrt(accm_density/(info.size()-1));

double result_stren=sqrt(accm_stren/(info.size()-1));

cout<<"强度方差"<<result_stren<<"密度方差"<<result_density<<"平均密度"<<density<<"平均强度"<<stren<<endl;

if(density>Thre1&&stren>Thre2&&result_density>Thre3&&result_stren>Thre4)
	return 1;
else 0;

}


int main(int argc, char* argv[])
{
	//视频不存在，就返回

	cv::VideoCapture cap(VideoFile);
	if(cap.isOpened()==false)
		return 0;

	//定义变量
	int i;
	feature_store.clear();
	cv::Mat frame;			//当前帧
	cv::Mat background;
	cv::Mat foreground;		//前景
	cv::Mat bw;				//中间二值变量
	cv::Mat se;				//形态学结构元素

	//用混合高斯模型训练背景图像
	cv::BackgroundSubtractorMOG2 mog;	
	for(i=0;i<TrainNo;++i)
	{
		cout<<"正在训练背景:"<<i<<endl;
		cap>>frame;
		//cvtColor(frame,frame,CV_RGB2GRAY);
		if(frame.empty()==true)
		{
			cout<<"视频帧太少，无法训练背景"<<endl;
			getchar();
			return 0;
		}
		mog(frame,foreground,0.01);	
	}

	//目标外接框、生成结构元素（用于连接断开的小目标）
	cv::Rect rt;
	se=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(5,5));

	//统计目标直方图时使用到的变量
	vector<cv::Mat> vecImg;
	vector<int> vecChannel;
	vector<int> vecHistSize;
	vector<float> vecRange;
	cv::Mat mask(frame.rows,frame.cols,cv::DataType<uchar>::type);
	//变量初始化
	vecChannel.push_back(0);
	vecHistSize.push_back(32);
	vecRange.push_back(0);
	vecRange.push_back(180);

	cv::Mat hsv;		//HSV颜色空间，在色调H上跟踪目标（camshift是基于颜色直方图的算法）
	cv::Mat back_hsv;//背景hsv空间
	cv::MatND hist;		//直方图数组
	double maxVal;		//直方图最大值，为了便于投影图显示，需要将直方图规一化到[0 255]区间上
	cv::Mat backP;		//反射投影图
	cv::Mat result;		//跟踪结果
	cv::Mat speed_angle;
	int frame_no=0;
	IplImage img_prev_grey;
	Mat grey,prev_grey;
	struct feature info;
	
	double density1=0;
	double count1=0;
	//视频处理流程
	while(1)
	{	
		//读视频
		cap>>frame;
		//cvtColor(frame,frame,CV_RGB2GRAY);
		if(frame.empty()==true)
			break;		

		//生成结果图
		frame.copyTo(result);
		frame.copyTo(speed_angle);
		memset(&info,0,sizeof(info));
		frame_no++;
		//检测目标前景
		mog(frame,foreground,0.01);
		//cout<<foreground.channels();
		//threshold(foreground, foreground, 128, 255, THRESH_BINARY_INV);
		//cv::imshow("混合高斯检测前景",foreground);
		//cvMoveWindow("混合高斯检测前景",400,0);
		
		///需要进行阴影检测
		
		mog.getBackgroundImage(background);
		//cout<<background.channels();
		cv::cvtColor(frame,hsv,cv::COLOR_BGR2HSV);
		cv::cvtColor(background,back_hsv,cv::COLOR_BGR2HSV);
		foreground=shade_detect(hsv,back_hsv,foreground);

		//对前景进行中值滤波、形态学膨胀操作，以去除伪目标和接连断开的小目标（一个大车辆有时会断开成几个小目标）	
		cv::medianBlur(foreground,foreground,5);
		//cv::imshow("中值滤波",foreground);
		//cvMoveWindow("中值滤波",800,0);
		cv::morphologyEx(foreground,foreground,cv::MORPH_DILATE,se);//形态学运算


		double density=den_cal(foreground);
		density1=density1+density;
		info.density=density;
		//检索前景中各个连通分量的轮廓
		foreground.copyTo(bw);
		vector<vector<cv::Point>> contours;
		cv::findContours(bw,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
		if(contours.size()<1)
			continue;
		//对连通分量进行排序
		std::sort(contours.begin(),contours.end(),biggerSort);

		//结合camshift更新跟踪位置（由于camshift算法在单一背景下，跟踪效果非常好；
		//但是在监控视频中，由于分辨率太低、视频质量太差、目标太大、目标颜色不够显著
		//等各种因素，导致跟踪效果非常差。  因此，需要边跟踪、边检测，如果跟踪不够好，
		//就用检测位置修改
		
		vecImg.clear();
		vecImg.push_back(hsv);
		for(int k=0;k<contours.size();++k)
		{
			//第k个连通分量的外接矩形框
			if(cv::contourArea(contours[k])<cv::contourArea(contours[0])/5)
				break;
			rt=cv::boundingRect(contours[k]);				
			mask=0;
			mask(rt)=255;

			//统计直方图
			cv::calcHist(vecImg,vecChannel,mask,hist,vecHistSize,vecRange);				
			cv::minMaxLoc(hist,0,&maxVal);
			hist=hist*255/maxVal;
			//计算反向投影图
			cv::calcBackProject(vecImg,vecChannel,hist,backP,vecRange,1);
			//camshift跟踪位置
			cv::Rect search=rt;
			cv::RotatedRect rrt=cv::CamShift(backP,search,cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,10,1));
			cv::Rect rt2=rrt.boundingRect();
			rt&=rt2;

			//跟踪框画到视频上
			cv::rectangle(result,rt,cv::Scalar(0,255,0),2);		


		}

		//结果显示
		//cv::imshow("Origin",frame);
		//cvMoveWindow("Origin",0,0);

		//cv::imshow("膨胀运算",foreground);
		//cvMoveWindow("膨胀运算",0,300);

		//cv::imshow("反向投影",backP);
		//cvMoveWindow("反向投影",400,300);
		
		
		
		cv::cvtColor(speed_angle,grey,COLOR_BGR2GRAY);
		grey.convertTo(grey,CV_8UC1);
		IplImage  img_grey= IplImage(grey); 
		CvSize winSize = cvSize(5,5);
		IplImage *velx = cvCreateImage( cvSize(grey.cols ,grey.rows),IPL_DEPTH_32F, 1 );
		IplImage *vely = cvCreateImage( cvSize(grey.cols ,grey.rows),IPL_DEPTH_32F, 1 );
		IplImage *abs_img= cvCreateImage( cvSize(grey.cols ,grey.rows),IPL_DEPTH_8U, 1 );
		if(frame_no!=1)
		{
		cvCalcOpticalFlowLK( &img_grey, &img_prev_grey, winSize, velx, vely );
		cvAbsDiff(&img_grey,&img_prev_grey, abs_img );
		
		CvScalar total_speed = cvSum(abs_img);
		int winsize=5;
		//cout<<total_speed.val[0];
		float ss = (float)total_speed.val[0]/(4*winsize*winsize)/255;
		//cout<<ss<<endl;
		CvScalar total_x = cvSum(velx);
		float xx = (float)total_x.val[0];

		CvScalar total_y = cvSum(vely);
		float yy = (float)total_y.val[0];

		double alpha_angle;

		if(xx<ZERO && xx>-ZERO)
			alpha_angle = PI/2;
		else
			alpha_angle = abs(atan(yy/xx));

		if(xx<0 && yy>0) alpha_angle = PI- alpha_angle ;
		if(xx<0 && yy<0) alpha_angle = PI + alpha_angle ;
		if(xx>0 && yy<0) alpha_angle = 2*PI - alpha_angle ;
		double count=sqrt(alpha_angle*alpha_angle+ss*ss);
		count1=count1+count;
		info.motion_strength=count;
		
		//cout<<count<<endl;

		}
		
		feature_store.push_back(info);
		grey.copyTo(prev_grey);
		img_prev_grey= IplImage(prev_grey); 
		
		if(frame_no%10==0)
		{
		//cout<<density1<<endl;
			//cout<<count1<<endl;
			int result=judge(feature_store,density1,count1);
		if (result==1)
			{
			Beep(500,500);//音乐
			}
		feature_store.clear();
		density1=0;
		count1=0;

		}
		cvReleaseImage(&velx);
		cvReleaseImage(&vely);
		cvReleaseImage(&abs_img);
		cv::imshow("跟踪效果",result);
		cvMoveWindow("跟踪效果",500,150);
		cvWaitKey(15);

	}

	getchar();
	return 0;
}

