#include "dataset/simpleCamera.hpp"


namespace stereo {

simpleCamera::simpleCamera()
{
	leftDetection=nullptr;
}


void simpleCamera::updateDetDesc(dataset::extractSettings in)
{
	std::string detectorName(in.ExtractorName.data);
	if(detectorName=="ORB")
	{
		int nmax=10000;
		float scale=float(in.orbConfig.scale.data);
		int level=int(in.orbConfig.level.data);
		int edge=int(in.orbConfig.edge.data);
		int wta=int(in.orbConfig.wta.data);
		int score=int(in.orbConfig.score.data);
		int patch=int(in.orbConfig.patch.data);
		
		leftDetection.release();
		rightDetection.release();
		leftDescriptor.release();
		rightDescriptor.release();
			
		leftDetection=cv::Ptr<DetectorSettings>(new OrbDetector(nmax,scale,level,edge,0,wta,score,patch));
		rightDetection=cv::Ptr<DetectorSettings>(new OrbDetector(nmax,scale,level,edge,0,wta,score,patch));
		leftDescriptor= cv::Ptr<DetectorSettings>(new OrbDetector(nmax,scale,level,edge,0,wta,score,patch));
		rightDescriptor=cv::Ptr<DetectorSettings>( new OrbDetector(nmax,scale,level,edge,0,wta,score,patch));
	}
	if(detectorName=="FAST")
	{
		int nmax=10000;
		float scale=float(in.orbConfig.scale.data);
		int level=int(in.orbConfig.level.data);
		int edge=int(in.orbConfig.edge.data);
		int wta=int(in.orbConfig.wta.data);
		int score=int(in.orbConfig.score.data);
		int patch=int(in.orbConfig.patch.data);
		
		int thresh=int(in.fastConfig.thresh.data);
		bool suppr=bool(in.fastConfig.suppression.data);
		int type=int(in.fastConfig.type.data);

		leftDetection.release();
		rightDetection.release();
		leftDescriptor.release();
		rightDescriptor.release();
		
		leftDetection=cv::Ptr<DetectorSettings>(new FastDetector(thresh,suppr,type));
		rightDetection=cv::Ptr<DetectorSettings>(new FastDetector(thresh,suppr,type));
		leftDescriptor= cv::Ptr<DetectorSettings>(new OrbDetector(nmax,scale,level,edge,0,wta,score,patch));
		rightDescriptor=cv::Ptr<DetectorSettings>( new OrbDetector(nmax,scale,level,edge,0,wta,score,patch));
		
	}

}

void simpleCamera::getROIimage(cv::Mat in, cv::Mat& out, bool left)
{
	cv::Mat tempOut;
	if(left)
	{
		tempOut=in(currentCam.l_ROI_);
		tempOut.copyTo(out);
		
	}
	else
	{
		tempOut=in(currentCam.r_ROI_);
		tempOut.copyTo(out);
	}

}

void simpleCamera::getImage(std::string in, cv::Mat& out, bool left, bool colour)
{
	cv::Mat fullImage=cv::imread(in,CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat tempOut;
	if(left)
	{
		tempOut=fullImage(cv::Rect(0,768,1024,768));
		cv::cvtColor(tempOut,tempOut,cv::COLOR_BayerBG2RGB);
		if(colour)
		{
			tempOut.copyTo(out);
		}
		else
		{
			cv::cvtColor(tempOut,tempOut, cv::COLOR_RGB2GRAY);
			tempOut.copyTo(out);
		}
		
	}
	else
	{
		tempOut=fullImage(cv::Rect(0,0,1024,768));
		cv::cvtColor(tempOut,tempOut,cv::COLOR_BayerBG2RGB);
		if(colour)
		{
			tempOut.copyTo(out);
		}
		else
		{
			cv::cvtColor(tempOut,tempOut, cv::COLOR_RGB2GRAY);
			tempOut.copyTo(out);
		}
	}


}

cv::Mat simpleCamera::getEpiDistanceMat(std::vector< cv::KeyPoint > left, std::vector< cv::KeyPoint > right)
{
	cv::Mat distances=cv::Mat(left.size(),right.size(),CV_64F);
	for(int leftIndex=0;leftIndex<left.size();leftIndex++)
	{
		for(int rightIndex=0;rightIndex<right.size();rightIndex++)
		{
			//offset left Distance
			cv::Point2f correctedLeft,correctedRight;
			correctedLeft.x=left.at(leftIndex).pt.x +currentCam.l_ROI_.x;
			correctedLeft.y=left.at(leftIndex).pt.y+currentCam.l_ROI_.y;
			
			correctedRight.x=right.at(rightIndex).pt.x +currentCam.r_ROI_.x;
			correctedRight.y=right.at(rightIndex).pt.y+currentCam.r_ROI_.y;
			
			distances.at<double>(leftIndex,rightIndex)=abs(correctedLeft.y-correctedRight.y);
			
		}
	}
	return distances;
}

cv::Mat simpleCamera::getEpiMask(std::vector< cv::KeyPoint > left, std::vector< cv::KeyPoint > right, cv::Mat distances,double threshold)
{
	cv::Mat mask=cv::Mat(left.size(),right.size(),CV_8U);
	for(int leftIndex=0;leftIndex<left.size();leftIndex++)
	{
		for(int index=0;index<right.size();index++)
		{
			if(distances.at<double>(leftIndex,index)<2)
			{
				mask.at<uchar>(leftIndex,index)=1;
			}
			else
			{
				mask.at<uchar>(leftIndex,index)=0;
			}
		}
	}
	return mask;
}


/*
std::vector< std::vector<int> > simpleCamera::filterEpiPole(std::vector<cv::KeyPoint> left,std::vector<cv::KeyPoint> right)
{
	
	std::vector< std::vector<int> > Ans;
	
	for(int leftIndex=0;leftIndex<left.size();leftIndex++)
	{
		std::vector<int> output;
		output.push_back(leftIndex);
		for(int index=0;index<right.size();index++)
		{
			if(distances.at<double>(leftIndex,index)<2)
			{
				output.push_back(index);
			}
		}
		Ans.push_back(output);
		
	}
	
	return Ans;
}
*/

void simpleCamera::undistort(cv::Mat in, cv::Mat& out, bool left)
{
	if(left)
	{
		out=cv::Mat(currentCam.L_fMapx_.size(),CV_8UC1);
		cv::remap(in,out,currentCam.L_fMapx_,currentCam.L_fMapy_,cv::INTER_LINEAR);
	}
	else
	{
		out=cv::Mat(currentCam.R_fMapx_.size(),CV_8UC1);
		cv::remap(in,out,currentCam.R_fMapx_,currentCam.R_fMapy_,cv::INTER_LINEAR);
	}
}


void simpleCamera::extractSettingsOne(std::string inImage)
{
	cv::Mat leftImage,rightImage;
	getImage(inImage,leftImage,true,false);
	getImage(inImage,rightImage,false,false);
	
	
}


void simpleCamera::drawROI(cv::Mat& in, cv::Mat& outImage, bool left)
{
	cv::cvtColor(in,outImage,cv::COLOR_GRAY2BGR);
	if(left)
	{
		cv::rectangle(outImage,currentCam.l_ROI_,cv::Scalar(0,0,255));
	}
	else
	{
		cv::rectangle(outImage,currentCam.r_ROI_,cv::Scalar(0,0,255));
	}

}

void simpleCamera::drawEpiLines(cv::Mat& in, cv::Mat& outImage, cv::Scalar col)
{
	if(in.channels()==1)
	{
		cv::cvtColor(in,outImage,cv::COLOR_GRAY2BGR);
	}
	else
	{
		in.copyTo(outImage);
	}
	
	for(int index=0;index<25;index++)
	{
		int step=in.size().height/25.0;
		cv::Point2f leftP(0,index*step);
		cv::Point2f rightP(in.size().width,index*step);
		cv::line(outImage,leftP,rightP,cv::Scalar(0,255,0));
	}
	
}


void simpleCamera::getSideSideRect(cv::Mat limg, cv::Mat rimg, cv::Mat& out)
{
	//rectifies a left and right image, and returns the two side by side
	out=cv::Mat(cv::Size(limg.size().width+rimg.size().width,limg.size().height),limg.type());
	cv::Mat lroi(out,cv::Rect(0,0,limg.size().width,limg.size().height));
	cv::Mat rroi(out,cv::Rect(limg.size().width,0,rimg.size().width,rimg.size().height));
	
	limg.copyTo(lroi);
	rimg.copyTo(rroi);
}



	
}
