 
#include <ros/console.h>
#include <ros/ros.h> 

#include "dataset/extractManager.hpp"
 
 int main(int argc, char **argv)
{
	using namespace stereo;
	ros::init(argc, argv, "extractNode");
	extractManager nodeMan;
	
	ros::spin();
	
	return 0;
}