#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>
#include <opencv2/opencv.hpp>

#include <iostream>
using namespace std;


int main(){
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device *dev = 0;
    libfreenect2::PacketPipeline *pipeline = 0;
    string serial = "";

    if(freenect2.enumerateDevices() == 0)
    {
        std::cout << "no device connected!" << std::endl;
        return -1;
    }
    if (serial == "")
    {
        serial = freenect2.getDefaultDeviceSerialNumber();
    }

    pipeline = new libfreenect2::CudaPacketPipeline();
    dev = freenect2.openDevice(serial, pipeline);

    int types = 0;
    bool enable_rgb = true; 
    bool enable_depth = true;

    if (enable_rgb)
        types |= libfreenect2::Frame::Color;
    if (enable_depth)
        types |= libfreenect2::Frame::Ir | libfreenect2::Frame::Depth;
    libfreenect2::SyncMultiFrameListener listener(types);
    libfreenect2::FrameMap frames;
    dev->setColorFrameListener(&listener);
    dev->setIrAndDepthFrameListener(&listener);

    if (enable_rgb && enable_depth)
    {
        if (!dev->start())
        return -1;
    }
    else
    {
        if (!dev->startStreams(enable_rgb, enable_depth))
        return -1;
    }
    
    std::cout << "device serial: " << dev->getSerialNumber() << std::endl;
    std::cout << "device firmware: " << dev->getFirmwareVersion() << std::endl;

    cout<< "RGB Camera Fx: " << dev->getColorCameraParams().fx << " Fy: " <<  dev->getColorCameraParams().fy << " cx: " << dev->getColorCameraParams().cx << " cy: " << dev->getColorCameraParams().cy <<endl;
    cout<< "Depth Camera Fx: " << dev->getIrCameraParams().fx << " Fy: " <<  dev->getIrCameraParams().fy << " cx: " << dev->getIrCameraParams().cx << " cy: " << dev->getIrCameraParams().cy <<endl;

    libfreenect2::Registration* registration = new libfreenect2::Registration(dev->getIrCameraParams(), dev->getColorCameraParams());
    libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4), bigdepth(1920,1082,4);

    bool protonect_shutdown=false;
    int framemax=10;
    int framecount=0;

    
    cv::Mat rgbmat, depthmat, depthmatUndistorted, rgbd, bigdepthmat;

    //while(!protonect_shutdown && (framemax == (size_t)-1 || framecount < framemax))
    while(true)
    {
        if (!listener.waitForNewFrame(frames, 10*1000)) // 10 sconds
        {
            std::cout << "timeout!" << std::endl;
            return -1;
        }
        libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
        //libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
        libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];
        cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data).copyTo(rgbmat);
        cv::Mat(depth->height, depth->width, CV_32FC1, depth->data).copyTo(depthmat);  
        
        registration->apply(rgb, depth, &undistorted, &registered, true, &bigdepth);
        listener.release(frames);

        cv::line(rgbmat,cv::Point2d((rgb->width/2)-20, rgb->height/2), cv::Point2d((rgb->width/2)+20, rgb->height/2), cv::Scalar(0));  //crosshair horizontal
        cv::line(rgbmat,cv::Point2d(rgb->width/2, (rgb->height/2)-20), cv::Point2d(rgb->width/2, (rgb->height/2)+20), cv::Scalar(0));  //crosshair vertical 
        
        cv::Mat(bigdepth.height, bigdepth.width, CV_32FC1, bigdepth.data).copyTo(bigdepthmat);
        cv::Mat(undistorted.height, undistorted.width, CV_32FC1, undistorted.data).copyTo(depthmatUndistorted);
        cv::Mat(registered.height, registered.width, CV_8UC4, registered.data).copyTo(rgbd); 
        
        //cv::imshow("rgb", rgbmat);
        //cv::imshow("depth", depthmat);
        cv::Mat visrgb, visdepth;
        cv::resize(rgbmat,visrgb,cv::Size(),0.5,0.5);
        cv::resize(bigdepthmat,visdepth,cv::Size(),0.5,0.5);

        //cv::hconcat(rgbmat,bigdepthmat,c);
        cv::imshow("rgb",visrgb );
        cv::imshow("bigdepth", visdepth/4500.0f);

        char key = cv::waitKey(1);
        
        if(key == ' '){
            depthmatUndistorted = depthmatUndistorted/1000; //depth in meter
            bigdepthmat = bigdepthmat/1000;
            cv::imwrite("../../snapshots/Depth" + to_string(framecount) + ".tiff", depthmatUndistorted);
            cv::imwrite("../../snapshots/DepthBig" + to_string(framecount) + ".tiff", bigdepthmat);
            cv::imwrite("../../snapshots/RGB" + to_string(framecount) + ".png", rgbmat);
            cv::imwrite("../../snapshots/RGBSmall" + to_string(framecount) + ".png", rgbd);
        }
        
        framecount++;
    }

    dev->stop();
    dev->close();
    return 0;
}
