#ifndef SIMPLE_CAMERA_HEADER_HPP
#define SIMPLE_CAMERA_HEADER_HPP

#include "Structures/CameraInfo/StereoRect.hpp"
#include "vSLAM_FrontEnd/Detection/OrbDetector.hpp"
#include "vSLAM_FrontEnd/Detection/FastDetector.hpp"
#include "vSLAM_FrontEnd/StereoFront/Matching.hpp"

#include <dataset/extractSettings.h>
#include <dataset/ORB.h>
#include <opencv2/imgproc.hpp>

namespace stereo
{

class simpleCamera
{
	public:
		StereoRect currentCam;
		cv::Ptr<DetectorSettings> leftDetection,rightDetection;
		cv::Ptr<DetectorSettings>  leftDescriptor,rightDescriptor;
		simpleCamera();
		void updateDetDesc(dataset::extractSettings in);
		void getROIimage(cv::Mat in,cv::Mat &out,bool left);
		void getImage(std::string in,cv::Mat &out,bool left,bool colour=false);
		cv::Mat getEpiDistanceMat(std::vector<cv::KeyPoint> left,std::vector<cv::KeyPoint> right);
		cv::Mat getEpiMask(std::vector<cv::KeyPoint> left,std::vector<cv::KeyPoint> right, cv::Mat distances,double threshold=2);
		void extractSettingsOne(std::string inImage);
		void undistort(cv::Mat in,cv::Mat &out,bool left);
	//	std::vector< std::vector<int> >filterEpiPole(std::vector<cv::KeyPoint> left,std::vector<cv::KeyPoint> right);
		void drawROI(cv::Mat &in,cv::Mat &outImage,bool left);
		void drawEpiLines(cv::Mat &in,cv::Mat &outImage,cv::Scalar col);
		void getSideSideRect(cv::Mat limg,cv::Mat rimg,cv::Mat &out);
		
		
		
};
}

#endif
