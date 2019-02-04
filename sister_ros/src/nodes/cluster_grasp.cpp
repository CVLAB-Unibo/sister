#include <iostream>
#include <time.h>

// ROS
#include <ros/ros.h>
#include <tf/transform_listener.h>

#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <atlas/viewer/Painter.hpp>

// Ros
ros::NodeHandle *nh;
tf::TransformListener *tf_listener;
ros::Publisher map_publisher;
ros::Publisher map_radius_publisher;

// Live parameters
std::string base_frame_name = "world";
std::string camera_frame_name = "camera";

typedef pcl::PointXYZRGBA PointType;

void drawSphere(Painter3D *&painter, Eigen::Matrix4f pose, Eigen::Vector3f color, std::string name)
{
    PointType p;
    p.x = pose(0, 3);
    p.y = pose(1, 3);
    p.z = pose(2, 3);
    painter->viewer->addSphere(p, 0.02, color(0), color(1), color(2), name);
}

void estimatePlane(pcl::PointCloud<PointType>::Ptr &cloud)
{
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<PointType> seg;
    // Optional
    seg.setOptimizeCoefficients(true);
    // Mandatory
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);

    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    std::cout << coefficients->values[0] << "," << coefficients->values[1] << "," << coefficients->values[2] << "," << coefficients->values[3] << "\n";
}

std::vector<pcl::PointCloud<PointType>::Ptr> extractCluster(pcl::PointCloud<PointType>::Ptr &cloud)
{
    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointType> ec;
    ec.setClusterTolerance(0.01); // 2cm
    ec.setMinClusterSize(250);
    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    std::cout << "Extraced: " << cluster_indices.size() << "\n";

    pcl::ExtractIndices<PointType> extract;

    extract.setInputCloud(cloud);

    std::vector<pcl::PointCloud<PointType>::Ptr> clusters;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        pcl::PointCloud<PointType>::Ptr cloud_cluster(new pcl::PointCloud<PointType>);
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
            cloud_cluster->points.push_back(cloud->points[*pit]); //*
        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
        clusters.push_back(cloud_cluster);
    }

    return clusters;
    //std::vector<pcl::PointCloud<PointType>::Ptr> clusters;
}
/**
 *
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv)
{

    // Initialize ROS
    ros::init(argc, argv, "cluster_grasp");
    nh = new ros::NodeHandle("~");
    //tf_listener = new tf::TransformListener();

    Painter3D *painter = new Painter3D();

    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
    pcl::io::loadPCDFile("/tmp/cloud.pcd", *cloud);

    Eigen::Matrix4f camera_pose = Eigen::Matrix4f::Zero();
    camera_pose(1, 0) = -1;
    camera_pose(0, 1) = -1;
    camera_pose(2, 2) = -1;
    camera_pose(3, 3) = 1;
    camera_pose(2, 3) = 0.52;

    drawSphere(painter, camera_pose, Eigen::Vector3f(0, 0, 1), "camera");
    //drawSphere(painter, Eigen::Matrix4f::Identity(), Eigen::Vector3f(1, 1, 1), "world");

    pcl::PointCloud<PointType>::Ptr world_cloud(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*cloud, *world_cloud, camera_pose);

    pcl::PointCloud<PointType>::Ptr filtered_cloud(new pcl::PointCloud<PointType>());

    float max_z = 0.01;

    for (int i = 0; i < world_cloud->points.size(); i++)
    {
        PointType p = world_cloud->points[i];
        if (p.z > max_z)
        {
            filtered_cloud->points.push_back(p);
        }
    }

    ROS_INFO_STREAM("Loaded" << cloud->points.size());

    estimatePlane(cloud);

    std::vector<pcl::PointCloud<PointType>::Ptr> clusters = extractCluster(filtered_cloud);
    std::cout << "EXTRACTED" << clusters.size() << "\n";

    srand(time(NULL));
    for (int i = 0; i < clusters.size(); i++)
    {

        pcl::PointCloud<PointType>::Ptr cluster = clusters[i];
        float r = rand() % 255;
        float g = rand() % 255;
        float b = rand() % 255;

        std::stringstream ss;
        ss << "cluster_" << i;
        painter->showCloud<PointType>(cluster, Eigen::Vector3f(r / 255., g / 255., b / 255.), 3, ss.str());
    }

    painter->showCloud<PointType>(world_cloud, Eigen::Vector3f(1, 1, 1), 1, "world_cloud");
    //painter->showCloud<PointType>(filtered_cloud, Eigen::Vector3f(1, 0, 1), 2, "filtered");

    // Frames
    // nh->param<std::string>("base_frame_name", base_frame_name, "camera");
    nh->param<std::string>("camera_frame_name", camera_frame_name, "camera_rf");

    int hz;
    nh->param<int>("hz", hz, 30);

    // Spin & Time
    ros::Rate r(hz);

    // Spin
    while (nh->ok())
    {
        painter->viewer->spinOnce();

        ros::spinOnce();
        r.sleep();
    }
}
