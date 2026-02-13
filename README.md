# Perception for Autonomous Systems

This repository contains exercises and projects for the **Perception for Autonomous Systems** course at the Technical University of Denmark. Here, I spent 13 weeks of doing computer vision and perception for autonomous systems.

## Course Overview

This course covers fundamental concepts in computer vision and perception techniques essential for autonomous systems including image processing, feature detection, 3D vision, point cloud processing and state estimation. The course progresses from basic image operations to advanced techniques in depth sensing, registration and filtering.

---

## Weekly Breakdown

### **Image Fundamentals**
**Topics:** Image I/O, Channel Structure, Image Manipulation

**Exercises:**
- **Exercise 1 - Image I/O and Channel Structure**
  - Load and display images using OpenCV
  - Understand BGR vs RGB channel ordering
  - Access pixel values and extract individual color channels
  - Display channel decomposition

- **Exercise 2 - Image Indexing and Manipulation**
  - Access and modify pixel data
  - Extract regions of interest (ROI)
  - Resize images (fixed size and dynamic scaling)
  - Rotate images using OpenCV and imutils
  - Apply Gaussian blur filters
  - Draw shapes (rectangles, circles, lines) and text on images

**Project:**
1. Segment apples from an apple tree image
2. Count the number of apples and compare to ground truth (26 apples)
3. Modify apple colors
4. Segment leaves from the image
5. Green screen removal and background replacement
6. Improve edge detection using erosion/dilation

---

### **Feature Detection and Matching**
**Topics:** Line Detection, Corner Detection, Feature Descriptors, Optical Flow

**Exercises:**

- **Exercise 1 - Hough Transform**
  - Detect lines in images using Hough transform
  - Detect circles in images
  - Apply probabilistic Hough transform
  - Understand parameter tuning for robust detection

- **Exercise 2 - Feature Detection**
  - Harris Corner Detection (`cv2.cornerHarris`)
  - Shi-Tomasi Feature Detection (`cv2.goodFeaturesToTrack`)
  - Detect and mark corner features in images
  - Compare corner detection methods

- **Exercise 3 - Feature Matching**
  - Scale-Invariant Feature Transform (SIFT) keypoint detection
  - SIFT descriptor computation
  - Feature matching between images using Brute Force Matcher (`cv2.BFMatcher`)
  - Visualize keypoint matches between two images
  - Identify object instances in complex scenes

- **Exercise 4 - Optical Flow**
  - Sparse optical flow using Lucas-Kanade method (`cv2.calcOpticalFlowPyrLK`)
  - Track feature movement between consecutive frames
  - Visualize motion vectors
  - Analyze object motion patterns

**Project:** Feature-based image analysis and tracking

---

### **Template Matching and Stereo Vision**
**Topics:** Template Matching, Stereo Vision, Depth Estimation

**Exercises:**

- **Exercise 1 - Template Matching**
  - Basic template matching techniques
  - Multi-scale template matching
  - Template matching with rotation invariance
  - Find objects at different scales in images
  - Visualize matching results with bounding boxes

- **Exercise 2 - Stereo Block Matching**
  - Load and preprocess stereo image pairs
  - Create disparity maps from stereo images
  - Stereo matching parameters tuning:
    - Number of disparities (`numDisparities`)
    - Block size (`blockSize`)
    - Minimum disparity
    - Disparity uniqueness ratio
  - Convert disparity to depth maps
  - 3D visualization of reconstructed scenes

**Project:** Stereo vision reconstruction and depth analysis

---

### **Camera Geometry and Calibration**
**Topics:** Camera Calibration, Epipolar Geometry, 3D Ranging

**Exercises:**

- **Exercise 1 - Monochrome Camera Calibration**
  - Camera calibration using checkerboard patterns
  - Intrinsic parameter estimation
  - Lens distortion correction
  - Camera matrix computation
  - Undistort images using calibration data

- **Exercise 2 - Epipolar Geometry**
  - Epipolar line constraint in stereo vision
  - SIFT feature detection and matching between stereo pairs
  - Find corresponding points in stereo images
  - Brute Force matching implementation
  - Compute fundamental matrix
  - Draw epipolar lines on image pairs
  - Understand camera geometry relationships

- **Exercise 3 - Ranging with RANSAC**
  - Convert LIDAR distance measurements to Cartesian coordinates
  - RANSAC-based line fitting for wall detection
  - Robust estimation in noisy sensor data
  - Distinguish inliers from outliers
  - Extract geometric information from point clouds

**Project:** Multi-view geometry and depth reconstruction

---

### **3D Point Clouds and Registration**
**Topics:** RGBD Imaging, Point Cloud Processing, ICP Algorithm

**Exercises:**

- **Exercise 1 - RGBD Point Cloud Creation**
  - Load RGBD image pairs (color + depth)
  - Create point clouds from RGB and depth data
  - Apply camera intrinsics for 3D reconstruction
  - Visualize point clouds
  - Understand RGBD sensor data structure

- **Exercise 2 - Local Registration with ICP**
  - Iterative Closest Point (ICP) algorithm
  - Point cloud registration and alignment
  - Transform estimation between point clouds
  - ICP implementation with convergence criteria
  - Register multiple point clouds
  - Evaluate registration accuracy
  - Combine multiple scans into a single model

**Dataset:** Redwood RGBD indoor dataset (400 images)

**Project:** Reconstruct scenes from RGBD sequences

---

### **Point Cloud Analysis and Segmentation**
**Topics:** Clustering, Segmentation, Dimensionality Reduction

**Exercises:**

- **Exercise 1 - Geometric Analysis**
  - Point cloud statistics and properties
  - Compute normal vectors
  - Detect edges and boundaries
  - Establish geometric relationships in point clouds

- **Exercise 2 - Point Cloud Clustering**
  - K-means clustering on 3D point data
  - Segment point clouds by spatial proximity
  - Cluster validation using metrics
  - Visualize segmented regions with color mapping
  - Hierarchical clustering alternatives
  - DBSCAN for density-based clustering

- **Exercise 3 - Ranging**
  - Extended LIDAR analysis
  - Multiple wall detection
  - Obstacle identification

**Project:** Segment and classify point cloud regions

---

### **State Estimation and Filtering**
**Topics:** Probabilistic State Estimation, Histogram Filters, Kalman Filters

**Exercises:**

- **Exercise 1 - Histogram Filter**
  - Discrete probability distributions for state estimation
  - Bayes filter implementation
  - Measurement updates and motion updates
  - Localization using sensor models
  - Multi-hypothesis tracking

- **Exercise 2 - Kalman Filter**
  - Continuous state representation with Gaussians
  - Linear Kalman filter implementation
  - Prediction step (motion model)
  - Measurement update step (sensor fusion)
  - Covariance matrix updates
  - Combine multiple Gaussian distributions for measurement updates
  - Track moving objects in video sequences
  - Extended Kalman Filter concepts

**Project:** Real-time object tracking and localization

---

### **Dimensionality Reduction**
**Topics:** Principal Component Analysis (PCA), Feature Extraction, Classification

**Exercises:**

- **Exercise 1 - Pre-processing and Feature Extraction**
  - Data normalization and standardization
  - Feature scaling for machine learning
  - Visualization of high-dimensional data

- **Exercise 2 - Principal Component Analysis (PCA)**
  - Reduce dimensionality from 4D to 2D using PCA
  - Apply to Iris dataset
  - Compute principal components and variance explained
  - Visualize reduced-dimensional data
  - Train classifier (Linear SVM) on reduced features
  - Compare performance before and after dimensionality reduction
  - Understand variance preservation in PCA

**Project:** Dimensionality reduction for classification

---

### **dvanced Perception Tasks**
**Topics:** Multi-view Geometry, Pose Estimation, Advanced Image Processing

**Exercises:**

- **Exercise 1 - Advanced Image Processing**
  - Work with pre-loaded image and landmark data (`.npy` files)
  - Multi-view image analysis
  - Landmark detection and tracking
  - 2D reference points and projections
  - Image sequences and temporal consistency

**Data Files:**
- 6 images (`img_0.npy` - `img_5.npy`)
- 5 landmark sets (`landmark_0.npy` - `landmark_4.npy`)
- 5 reference 2D projections (`reference_2D_0.npy` - `reference_2D_4.npy`)

**Project:** Multi-view reconstruction and pose estimation
