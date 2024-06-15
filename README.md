

This project explores the analysis of point clouds using the [KITTI dataset](https://www.cvlibs.net/datasets/kitti/). Initially, we leverage  ```Open3D``` for visualization purposes and use  **Voxel Grid** downsampling to simplify the point clouds. Subsequently, we implement the **RANSAC** algorithm to **segment** obstacles from the road, improving our understanding of the scene. We further enhance spatial analysis by clustering similar obstacles using  **DBSCAN**.  For **tracking** purposes, **3D bounding boxes** are constructed around each identified obstacle.


### Visualise the point clouds
Utilized the KITTI Dataset, which comprises synchronous stereo images and LiDAR data. This dataset contains two pairs of images captured at consistent intervals through its stereo system. Moreover, the Velodyne LiDAR system produces a corresponding point cloud. Consequently, this allows us to visualize the scene in 2D using the images and in 3D with the point cloud, as illustrated below:
 <img width="956" alt="pcp2" src="https://github.com/AnoushkaBaidya/3D-Point-Cloud-Object-Segmentation-Pipeline/assets/115124698/44f2a01d-ffd9-4e90-8b27-46b27ad15f35">

In this project, there are several libraries and software available for point cloud processing. Among them, two commonly used libraries are **Open3D** and **PCL**.  The reason for this choice is its user-friendly nature and the abundant literature available on it. Our point cloud has allows us to employ the ```read_point_cloud function``` from Open3D as follows:

```python
point_cloud = open3d.io.read_point_cloud(point_cloud_path)
```

After reading the file, there are numerous ways to visualize the point cloud with Open3D:

```python
    o3d.visualization.draw_geometries([point_cloud])
```
![loaded_pcl](https://github.com/AnoushkaBaidya/3D-Point-Cloud-Object-Segmentation-Pipeline/assets/115124698/46ec477f-ba61-4078-831b-ca242ec15173)

<img width="905" alt="1" src="https://github.com/AnoushkaBaidya/3D-Point-Cloud-Object-Segmentation-Pipeline/assets/115124698/47978fc9-36ad-4ce7-9d42-77ca12d22d10">



### Downsample the point clouds

It's important to note that when processing our point cloud, it isn't essential to use every point. In fact, downsampling the point cloud can be beneficial for removing potential noise and achieving faster processing and visualization. The steps for Voxel Grid Downsampling are as follows:

1. **Define the Voxel Size**: Smaller voxel sizes will result in a more detailed point cloud, while larger voxel sizes will lead to more aggressive downsampling.
2. **Divide the Space**: The 3D space of the point cloud is divided into voxels by creating a 3D grid. Each point in the point cloud is assigned to the voxel it falls into based on its 3D coordinates.
3. **Subsampling**: For each voxel, one point is retained. This can be achieved by selecting the ```centroid``` of the points within the voxel or by using other sampling methods.
4. **Remove Redundant Points**: Points that fall into the same voxel are reduced to one point, effectively downsampling the point cloud.

 
```python
    # Voxel Grid
    point_cloud_downsampled = point_cloud.voxel_down_sample(voxel_size=1)
```

The downsampling process results in a point cloud with a reduced number of points while still retaining its main features.

![downsample_pcl](https://github.com/AnoushkaBaidya/3D-Point-Cloud-Object-Segmentation-Pipeline/assets/115124698/20043d5d-afdf-4daf-812b-8c2194bf9929)

### Segment the point clouds as inliers (objects of interest) and outliers (ground)
RANSAC (```Random Sample Consensus```) was invented by Fischler and Bolles in ```1981``` as a solution to the problem of fitting models (such as lines, planes, circles, etc.) to noisy data with outliers. The algorithm is widely used in computer vision, image processing, and other fields where robust estimation of model parameters is required.

1. Randomly choose ```s``` samples which is the minimum number of samples to fit a model.
2. Fit the model to the randomly chosen samples.
3. Count the number ```M``` of datapoints which fit the model within a measure of error ```e```.
4. Repeat steps 1-3 ```N``` times.
5. Choose the model that has the highest number of ```M``` inliers.


```python
    # Perform plane segmentation using RANSAC
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)

    # Extract inlier and outlier point clouds
    inlier_cloud = point_cloud.select_by_index(inliers)
    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)
```

![3](https://github.com/AnoushkaBaidya/3D-Point-Cloud-Object-Segmentation-Pipeline/assets/115124698/12cb1999-7e0c-41fb-a189-3c7be377e70b)


### Clustering the inliers to individual objects

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm proposed by Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu in 1996. It is particularly suitable for datasets with complex structures and a need for automatic cluster detection. 

1. Choose parameters ```ε``` and ```MinPts``` that define the **neighborhood** of a data point and the **minimum points** for forming core points, respectively.
2. Mark data points as **core points** if they have at least MinPts points within ε-neighborhood.
3. A point A is directly **density-reachable** from B if B is core and A is within its ε-neighborhood.
4. Density-reachability is **transitive**, forming density-based clusters by linking mutually density-reachable points.
5. ```Noise points``` are **outliers**, not core, and not density-reachable from any core point.


```python
    labels = np.array(outlier_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
```


![image](https://github.com/AnoushkaBaidya/3D-Point-Cloud-Object-Segmentation-Pipeline/assets/115124698/f12eab5d-ce51-43de-af57-8f45fabe076f)


### Compute 3D bounding box for each cluster
After successfully clustering our obstacles, the next step is to enclose them within 3D bounding boxes. Given that we are working in a 3D space, the bounding boxes we create are also in 3D. However, it's essential to recognize that in addition to generating bounding boxes, we require information about their **orientation**. As some obstacles might be positioned at various angles relative to the ego vehicle, we aim to construct bounding boxes that precisely encompass each obstacle. We will use ```PCA``` for that:

1. Compute the **centroid** of the point cloud subset, which is the center of mass.

2. Calculate the **covariance matrix** for the subset, summarizing dimension relationships.

3. Apply **PCA** to the covariance matrix, finding **eigenvectors** and **eigenvalues**.

4. The eigenvector with the **largest** eigenvalue signifies the longest axis and primary **spread direction**.

5. The eigenvector with the **smallest** eigenvalue indicates the shortest axis and **compression direction**.

6. Use these eigenvectors to define the **orientation** (```rotation```) of the **Oriented Bounding Box** (```OBB```). It aligns the OBB with the eigenvectors, so the box represents the point cloud's **principal directions**.

7. OBB **size** along each axis reflects point **spread** along the corresponding eigenvector.

```python
    obs = []
    # Group points by cluster label
    indexes = pd.Series(range(len(labels))).groupby(labels, sort=False).apply(list).tolist()

    # Iterate over clusters and perform PCA
    for i in range(0, len(indexes)):
        nb_points = len(outlier_cloud.select_by_index(indexes[i]).points)
        if nb_points > MIN_POINTS and nb_points < MAX_POINTS:
            sub_cloud = outlier_cloud.select_by_index(indexes[i])
            obb = sub_cloud.get_oriented_bounding_box()
```

Note that the ```get_oriented_bounding_box``` function from Open3D performs the PCA function for us behind the scenes. We use the outlier point cloud subset as these correspond to the cars and other objects whereas the inliers represent the road.

![image](https://github.com/AnoushkaBaidya/3D-Point-Cloud-Object-Segmentation-Pipeline/assets/115124698/a7046923-0d3f-4b8c-9de7-32892d8d46f9)


## Final Result
![visualization](https://github.com/AnoushkaBaidya/3D-Point-Cloud-Object-Segmentation-Pipeline/assets/115124698/3c7b1311-5130-4b5b-aa3f-3e4f2cec1e3d)

