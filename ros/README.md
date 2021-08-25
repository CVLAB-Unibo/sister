# Sister and ROS Service example

First you need a simple service with the `5 images` in the input request along with the `disp_count` parameter. The response should be a `single image` representing the computed multiview disparity. A prototype example could be:


```
int32 disp_count
sensor_msgs/Image center
sensor_msgs/Image left
sensor_msgs/Image top
sensor_msgs/Image right
sensor_msgs/Image bottom
---
sensor_msgs/Image disparity
```


The service should have a single callback receiving the 5 images and exploiting the `SisterMultiviewDisparities` helper class to computer the multiview disparity. The (pseudo)complex part is only the OpenCV->ROS conversion (and back). A prototype callback implementation should be:

```c++

#include "iostream"
#include "opencv2/opencv.hpp"
#include <sister/SisterMultiviewDisparities.hpp>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

void (
    sister::multiview_server::Request  &request,
    sister::multiview_server::Response &res
){

    // OpenCV pointer declaration
    cv_bridge::CvImagePtr cv_center, cv_left, cv_top, cv_right, cv_bottom;

    // ROS to OpenCV
    cv_center = cv_bridge::toCvCopy(request.center, sensor_msgs::image_encodings::BGR8);
    cv_left = cv_bridge::toCvCopy(request.left, sensor_msgs::image_encodings::BGR8);
    cv_top = cv_bridge::toCvCopy(request.top, sensor_msgs::image_encodings::BGR8);
    cv_right = cv_bridge::toCvCopy(request.right, sensor_msgs::image_encodings::BGR8);
    cv_bottom = cv_bridge::toCvCopy(request.bottom, sensor_msgs::image_encodings::BGR8);

    // Multiview matcher
    SisterMultiviewDisparities multiview(
		*cv_center,
		*cv_right,
		*cv_top,
		*cv_left,
		*cv_bottom);

	// Compute disparities
	cv::Mat result_horizontal, result_vertical, result_multiview;
	multiview.compute_disparities(request.disp_count, result_multiview, result_horizontal, result_vertical);

	// Colorize disparities
	
	cv::normalize(result_multiview, result_multiview, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::applyColorMap(result_multiview, result_multiview, cv::COLORMAP_MAGMA);

    // OpenCV to ROS
    sensor_msgs::ImagePtr output_disparity = cv_bridge::CvImage(std_msgs::Header(), "bgr8", result_multiview).toImageMsg();

    // Build response
    res.disparity = output_disparity;
}

```