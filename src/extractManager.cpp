#include "dataset/extractManager.hpp"

namespace stereo {
	
extractManager::extractManager()
{
	extractServ=n.advertiseService("extract",&extractManager::extract,this);
	
	updateServ=n.advertiseService("extract/settings",&extractManager::update,this);
	rectServ=n.advertiseService("extract/rectified",&extractManager::updateRect,this);
	stereoServ=n.advertiseService("extract/stereo",&extractManager::stereoExtract,this);
	extractFASTserv=n.advertiseService("extract/FAST",&extractManager::extractFAST,this);
	extractSIFTserv=n.advertiseService("extract/SIFT",&extractManager::extractSIFT,this);
	extractORBserv=n.advertiseService("extract/ORB",&extractManager::extractORB,this);
	extractSURFserv=n.advertiseService("extract/SURF",&extractManager::extractSURF,this);
	extractAKAZEserv=n.advertiseService("extract/AKAZE",&extractManager::extractAKAZE,this);
	extractBRISKserv=n.advertiseService("extract/BRISK",&extractManager::extractBRISK,this);
	
}


bool extractManager::extractBRISK(dataset::extractBRISK::Request& req, dataset::extractBRISK::Response& res)
{
	//BRISK::BRISK(int thresh=30, int octaves=3, float patternScale=1.0f)¶
	
	
	int n = int(req.config.thresh.data);
	int oct = int(req.config.octaves.data);
	float pattern=float(req.config.patternScale.data);
	
	//get the input images
	
	cv::Mat leftImage,rightImage;
	bumble.getImage(std::string(req.imageDir.data),leftImage,true);
	bumble.getImage(std::string(req.imageDir.data),rightImage,false);
	
	//undistort images
	cv::Mat leftundist,rightundist;
	bumble.undistort(leftImage,leftundist,true);
	bumble.undistort(rightImage,rightundist,false);
	//get ROI images
	cv::Mat lr,rr;
	bumble.getROIimage(leftundist,lr,true);
	bumble.getROIimage(rightundist,rr,false);
	
	double average=0;
	
	cv::BRISK det(n,oct,pattern);
	for(int index=0;index<5;index++)
	{
		std::vector<cv::KeyPoint> lf;
		auto start=std::chrono::steady_clock::now();
		det(lr,cv::Mat(),lf);
		auto end=std::chrono::steady_clock::now();
		average+= std::chrono::duration<double,std::milli>(end-start).count();
	}
	

	average= average/5.0;
	//extractfeatures
	std::vector<cv::KeyPoint> leftF,rightF;
	det(lr,cv::Mat(),leftF);
//	//draw features onto image, and save it to a directory
	
	if(std::string(req.outputDir.data)!="")
	{
		cv::Mat featl;
		cv::drawKeypoints(lr,leftF,featl);
		cv::imwrite(std::string(req.outputDir.data),featl);
	}
	res.nleft.data=leftF.size();
	res.averageTime.data=average;
	return true;
}

bool extractManager::extractSIFT(dataset::extractSIFT::Request& req, dataset::extractSIFT::Response& res)
{
	int n=int(req.config.maxFeatures.data);
	int oct=int(req.config.nOctave.data);
	double contrast=double(req.config.contrastThresh.data);
	double edge=double(req.config.edgeThresh.data);
	double sigma=double(req.config.sigma.data);
	
	cv::SIFT siftDetector= cv::SIFT(n,oct,contrast,edge,sigma);
	
	//get the input images
	
		cv::Mat leftImage,rightImage;
	bumble.getImage(std::string(req.imageDir.data),leftImage,true);
	bumble.getImage(std::string(req.imageDir.data),rightImage,false);
	
	//undistort images
	cv::Mat leftundist,rightundist;
	bumble.undistort(leftImage,leftundist,true);
	bumble.undistort(rightImage,rightundist,false);
	//get ROI images
	cv::Mat lr,rr;
	bumble.getROIimage(leftundist,lr,true);
	bumble.getROIimage(rightundist,rr,false);
	
	double average=0;
	for(int index=0;index<5;index++)
	{
		std::vector<cv::KeyPoint> lf;
		auto start=std::chrono::steady_clock::now();
		siftDetector(lr,cv::Mat(),lf);
		auto end=std::chrono::steady_clock::now();
		average+= std::chrono::duration<double,std::milli>(end-start).count();
	}
	

	average= average/5.0;
	//extractfeatures
	std::vector<cv::KeyPoint> leftF,rightF;
	siftDetector(lr,cv::Mat(),leftF);
//	//draw features onto image, and save it to a directory
	
	if(std::string(req.outputDir.data)!="")
	{
		cv::Mat featl;
		cv::drawKeypoints(lr,leftF,featl);
		cv::imwrite(std::string(req.outputDir.data),featl);
	}
	res.nleft.data=leftF.size();
	res.averageTime.data=average;
	return true;
}


bool extractManager::extractFAST(dataset::extractFAST::Request& req, dataset::extractFAST::Response& res)
{
	//create a fast detector object
	int thresh=int(req.config.thresh.data);
	bool suppr=bool(req.config.suppression.data);
	int type=int(req.config.type.data);
	
	
	bumble.leftDetection.release();
	bumble.rightDetection.release();
	
	bumble.leftDetection=cv::Ptr<DetectorSettings>(new FastDetector(thresh,suppr,type));
	bumble.rightDetection=cv::Ptr<DetectorSettings>(new FastDetector(thresh,suppr,type));
	
	//get the input images
	
	
	
	cv::Mat leftImage,rightImage;
	bumble.getImage(std::string(req.imageDir.data),leftImage,true);
	bumble.getImage(std::string(req.imageDir.data),rightImage,false);
	
	//undistort images
	cv::Mat leftundist,rightundist;
	bumble.undistort(leftImage,leftundist,true);
	bumble.undistort(rightImage,rightundist,false);
	//get ROI images
	cv::Mat lr,rr;
	bumble.getROIimage(leftundist,lr,true);
	bumble.getROIimage(rightundist,rr,false);
	
	double average=0;
	for(int index=0;index<5;index++)
	{
		//extractfeatures
		std::vector<cv::KeyPoint> lf;
		auto start=std::chrono::steady_clock::now();
		bumble.leftDetection->detect(lr,lf);
		auto end=std::chrono::steady_clock::now();
		average+= std::chrono::duration<double,std::milli>(end-start).count();
	}

		
	average= average/5.0;
	//extractfeatures
	std::vector<cv::KeyPoint> leftF;
	bumble.leftDetection->detect(lr,leftF);
	//draw features onto image, and save it to a directory
	if(std::string(req.outputDir.data)!="")
	{
		cv::Mat featl;
		cv::drawKeypoints(lr,leftF,featl);
		cv::imwrite(std::string(req.outputDir.data),featl);
	}
	
	
	res.nleft.data=leftF.size();
	res.averageTime.data=average;
	return true;
}



bool extractManager::extractAKAZE(dataset::extractAKAZE::Request& req, dataset::extractAKAZE::Response& res)
{
	AKAZEOptions settings;
	
	settings.soffset=int(req.config.offset.data);
	settings.derivative_factor=float(req.config.derivative_factor.data);
	settings.omax=int(req.config.omax.data);
	settings.nsublevels=int(req.config.sublevels.data);
	settings.dthreshold=float(req.config.dthreshold.data);
	settings.min_dthreshold=float(req.config.minthresh.data);
	settings.diffusivity=static_cast<DIFFUSIVITY_TYPE>(req.config.diffusivity.data);
	settings.descriptor=static_cast<DESCRIPTOR_TYPE>(req.config.descriptor.data);
	settings.descriptor_channels=1;//assume grey image
	settings.descriptor_pattern_size=int(req.config.patternSize.data);
	settings.sderivatives=float(req.config.sderivative.data);
	settings.kcontrast=float(req.config.kcontrast.data);
	settings.kcontrast_percentile=float(req.config.kcontrast_percentile.data);
	settings.kcontrast_nbins=size_t(req.config.kcontrast_nbins.data);
	settings.show_results=false;
	//get the input images
	
	cv::Mat leftImage,rightImage;
	bumble.getImage(std::string(req.imageDir.data),leftImage,true);
	bumble.getImage(std::string(req.imageDir.data),rightImage,false);
	
	//undistort images
	cv::Mat leftundist,rightundist;
	bumble.undistort(leftImage,leftundist,true);
	bumble.undistort(rightImage,rightundist,false);
	//get ROI images
	cv::Mat lr,rr;
	bumble.getROIimage(leftundist,lr,true);
	bumble.getROIimage(rightundist,rr,false);
	
	settings.img_width=lr.cols;
	settings.img_height=lr.rows;
	AkazeDetector a(settings);

	double average=0;
	for(int index=0;index<5;index++)
	{
		std::vector<cv::KeyPoint> lf;
		auto start=std::chrono::steady_clock::now();
		a.detect(lr,lf);
		auto end=std::chrono::steady_clock::now();
		average+= std::chrono::duration<double,std::milli>(end-start).count();
	}
	

	average= average/5.0;
	//extractfeatures
	std::vector<cv::KeyPoint> leftF,rightF;
	a.detect(lr,leftF);
//	//draw features onto image, and save it to a directory
	
	if(std::string(req.outputDir.data)!="")
	{
		cv::Mat featl;
		cv::drawKeypoints(lr,leftF,featl);
		cv::imwrite(std::string(req.outputDir.data),featl);
	}
	res.nleft.data=leftF.size();
	res.averageTime.data=average;
	return true;
}

bool extractManager::extractORB(dataset::extractORB::Request& req, dataset::extractORB::Response& res)
{
	int n=int(req.config.maxFeatures.data);
	double scale=double(req.config.scale.data);
	int edge =int(req.config.edge.data);
	int level = int(req.config.level.data);
	int wta=int(req.config.wta.data);
	int score=int(req.config.score.data);
	int patch=int(req.config.patch.data);
	
	cv::ORB OrbDetector= cv::ORB(n,scale,level,edge,0,wta,score,patch);
	
	//get the input images
	
	cv::Mat leftImage,rightImage;
	bumble.getImage(std::string(req.imageDir.data),leftImage,true);
	bumble.getImage(std::string(req.imageDir.data),rightImage,false);
	
	//undistort images
	cv::Mat leftundist,rightundist;
	bumble.undistort(leftImage,leftundist,true);
	bumble.undistort(rightImage,rightundist,false);
	//get ROI images
	cv::Mat lr,rr;
	bumble.getROIimage(leftundist,lr,true);
	bumble.getROIimage(rightundist,rr,false);
	
	double average=0;
	for(int index=0;index<5;index++)
	{
		std::vector<cv::KeyPoint> lf;
		auto start=std::chrono::steady_clock::now();
		OrbDetector(lr,cv::Mat(),lf);
		auto end=std::chrono::steady_clock::now();
		average+= std::chrono::duration<double,std::milli>(end-start).count();
	}
	

	average= average/5.0;
	//extractfeatures
	std::vector<cv::KeyPoint> leftF,rightF;
	OrbDetector(lr,cv::Mat(),leftF);
//	//draw features onto image, and save it to a directory
	
	if(std::string(req.outputDir.data)!="")
	{
		cv::Mat featl;
		cv::drawKeypoints(lr,leftF,featl);
		cv::imwrite(std::string(req.outputDir.data),featl);
	}
	res.nleft.data=leftF.size();
	res.averageTime.data=average;
	return true;
}


bool extractManager::extractSURF(dataset::extractSURF::Request &req,dataset::extractSURF::Response &res)
{
	
	double thresh=double(req.config.thresh.data);
	int nOctaves=int(req.config.nOctave.data);
	int nOctaveLayer=int(req.config.nOctaveLayer.data);
	bool extended=bool(req.config.extended.data);
	bool upright=bool(req.config.upright.data);

	cv::SURF SurfDetector= cv::SURF(thresh,nOctaves,nOctaveLayer,extended,upright);
	
	//get the input images
	
	cv::Mat leftImage,rightImage;
	bumble.getImage(std::string(req.imageDir.data),leftImage,true);
	bumble.getImage(std::string(req.imageDir.data),rightImage,false);
	
	//undistort images
	cv::Mat leftundist,rightundist;
	bumble.undistort(leftImage,leftundist,true);
	bumble.undistort(rightImage,rightundist,false);
	//get ROI images
	cv::Mat lr,rr;
	bumble.getROIimage(leftundist,lr,true);
	bumble.getROIimage(rightundist,rr,false);
	
	double average=0;
	for(int index=0;index<5;index++)
	{
		std::vector<cv::KeyPoint> lf;
		auto start=std::chrono::steady_clock::now();
		SurfDetector(lr,cv::Mat(),lf);
		auto end=std::chrono::steady_clock::now();
		average+= std::chrono::duration<double,std::milli>(end-start).count();
	}
	

	average= average/5.0;
	//extractfeatures
	std::vector<cv::KeyPoint> leftF,rightF;
	SurfDetector(lr,cv::Mat(),leftF);
//	//draw features onto image, and save it to a directory
	
	if(std::string(req.outputDir.data)!="")
	{
		cv::Mat featl;
		cv::drawKeypoints(lr,leftF,featl);
		cv::imwrite(std::string(req.outputDir.data),featl);
	}
	res.nleft.data=leftF.size();
	res.averageTime.data=average;
	return true;
	
}



bool extractManager::update(dataset::updateSettings::Request& req, dataset::updateSettings::Response& res)
{
	bumble.updateDetDesc(req.newSett);
	return true;
}

bool extractManager::updateRect(dataset::rectifiedSettings::Request& req, dataset::rectifiedSettings::Response& res)
{
	cv::FileStorage in(std::string(req.RectifiedXMLdir.data),cv::FileStorage::READ);
	in["StereoRect"]>>bumble.currentCam;
	in.release();
	return true;
}


bool extractManager::extract(dataset::extractFeatures::Request& req, dataset::extractFeatures::Response& res)
{
	//get input images
	cv::Mat leftImage,rightImage;
	bumble.getImage(std::string(req.imageDir.data),leftImage,true);
	bumble.getImage(std::string(req.imageDir.data),rightImage,false);
	
	//undistort images
	cv::Mat leftundist,rightundist;
	bumble.undistort(leftImage,leftundist,true);
	bumble.undistort(rightImage,rightundist,false);
	//get ROI images
	cv::Mat lr,rr;
	bumble.getROIimage(leftundist,lr,true);
	bumble.getROIimage(rightundist,rr,false);
	
	//extractfeatures
	std::vector<cv::KeyPoint> leftF,rightF;
	bumble.leftDetection->detect(lr,leftF);
	bumble.rightDetection->detect(rr,rightF);
	//populate response
	res.nleft.data=leftF.size();
	res.nright.data=rightF.size();
	res.leftROIDir.data=std::string(DEFAULT_SAVE_DIRECTORY)+"/leftROI.ppm";
	res.rightROIDir.data=std::string(DEFAULT_SAVE_DIRECTORY)+"/rightROI.ppm";
	res.leftFoundDir.data=std::string(DEFAULT_SAVE_DIRECTORY)+"/featLeft.ppm";
	res.rightFoundDir.data=std::string(DEFAULT_SAVE_DIRECTORY)+"/featRight.ppm";
	//make ROI image 
	cv::Mat outROIl,outROIr;
	bumble.drawROI(leftundist,outROIl,true);
	bumble.drawROI(rightundist,outROIr,false);
	//make features image 
	cv::Mat featl,featr;
	cv::drawKeypoints(lr,leftF,featl);
	cv::drawKeypoints(rr,rightF,featr);
	//save them
		//make file names 
	cv::imwrite(std::string(DEFAULT_SAVE_DIRECTORY)+"/leftROI.ppm",outROIl);
	cv::imwrite(std::string(DEFAULT_SAVE_DIRECTORY)+"/rightROI.ppm",outROIr);
	cv::imwrite(std::string(DEFAULT_SAVE_DIRECTORY)+"/featLeft.ppm",featl);
	cv::imwrite(std::string(DEFAULT_SAVE_DIRECTORY)+"/featRight.ppm",featr);
	

	return true;
}

bool extractManager::stereoExtract(dataset::stereoInliers::Request& req, dataset::stereoInliers::Response& res)
{
		//get input images
	cv::Mat leftImage,rightImage;
	bumble.getImage(std::string(req.imageDir.data),leftImage,true);
	bumble.getImage(std::string(req.imageDir.data),rightImage,false);
	
	//undistort images
	cv::Mat leftundist,rightundist;
	bumble.undistort(leftImage,leftundist,true);
	bumble.undistort(rightImage,rightundist,false);
	//get ROI images
	cv::Mat lr,rr;
	bumble.getROIimage(leftundist,lr,true);
	bumble.getROIimage(rightundist,rr,false);
	
	//extractfeatures
	std::vector<cv::KeyPoint> leftF,rightF;
	bumble.leftDetection->detect(lr,leftF);
	bumble.rightDetection->detect(rr,rightF);
	
	res.nleft.data=leftF.size();
	res.nright.data=rightF.size();
	

	
	cv::Mat leftD,rightD;
	bumble.leftDescriptor->extract(lr,leftF,leftD);
	bumble.rightDescriptor->extract(rr,rightF,rightD);
	
	cv::Mat distances=bumble.getEpiDistanceMat(leftF,rightF);
	cv::Mat epiMask=bumble.getEpiMask(leftF,rightF,distances);
	std::vector<std::vector<cv::DMatch>> out;
	
	
	
	cv::BFMatcher m(cv::NORM_HAMMING,false);
	m.knnMatch(leftD, rightD,out,2,epiMask);
	
	std::vector<cv::DMatch> good;
	for(int index=0;index<out.size();index++)
	{
		if(out.at(index).size()>=2)
		{
			if(out.at(index).at(0).distance<0.8*out.at(index).at(1).distance)
			{
				good.push_back(out.at(index).at(0));
			}
		}
		else
		{
			if(out.at(index).size()==1)
			{
				good.push_back(out.at(index).at(0));
			}
		}
		
	}

	
	cv::Mat ff;
	
	cv::drawMatches(lr,leftF, rr, rightF,good,ff,cv::Scalar::all(-1),cv::Scalar::all(-1),std::vector<char>(),2);
	
	res.nInliers.data=good.size();

	cv::imwrite(std::string(DEFAULT_SAVE_DIRECTORY)+"/matchedImage.ppm",ff);
	res.matchedDir.data=std::string(DEFAULT_SAVE_DIRECTORY)+"/matchedImage.ppm";
	
	return true;
}



	
}
