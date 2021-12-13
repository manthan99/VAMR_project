%% Setup
clear all;
close all;
clc;

ds = 0; % 0: KITTI, 1: Malaga, 2: parking

if ds == 0
    % need to set kitti_path to folder containing "05" and "poses"
    kitti_path = '../data/kitti05/kitti';
    %assert(exist(kitti_path, 'var') ~= 0);
    ground_truth = load([kitti_path '/poses/05.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    last_frame = 4540;
    K = [7.188560000000e+02 0 6.071928000000e+02
        0 7.188560000000e+02 1.852157000000e+02
        0 0 1];
elseif ds == 1
    % Path containing the many files of Malaga 7.
    malaga_path = 'malaga-urban-dataset-extract-07';
    %assert(exist('malaga_path', 'var') ~= 0);
    images = dir([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    left_images = images(3:2:end);
    last_frame = length(left_images);
    K = [621.18428 0 404.0076
        0 621.18428 309.05989
        0 0 1];
elseif ds == 2
    % Path containing images, depths and all...
    parking_path = 'parking';
    %assert(exist('parking_path', 'var') ~= 0);
    last_frame = 598;
    K = load([parking_path '/K.txt']);
     
    ground_truth = load([parking_path '/poses.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
else
    assert(false);
end

%% Bootstrap: Select and Import Frames

%Frames used for initialization
bootstrap_frames = [0, 4]; % 115 125

if ds == 0
    img0 = imread([kitti_path '/05/image_0/' ...
        sprintf('%06d.png',bootstrap_frames(1))]);
    img1 = imread([kitti_path '/05/image_0/' ...
        sprintf('%06d.png',bootstrap_frames(2))]);
elseif ds == 1
    img0 = rgb2gray(imread([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(1)).name]));
    img1 = rgb2gray(imread([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(2)).name]));
elseif ds == 2
    img0 = rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png',bootstrap_frames(1))]));
    img1 = rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png',bootstrap_frames(2))]));
else
    assert(false);
end

%% Bootstrap: Detect and match features and estimate pose
% Note all arguments that were altered from their default are marked with a
% comment

% Detect corners using FAST features
corners = detectFASTFeatures(img0,...
    'MinQuality',0.1,...
    'MinContrast',0.2,...
    'ROI',[1 1 size(img0,2) size(img0,1)]);

% Initialize tracker object for KLT based point tracking
pointTracker = vision.PointTracker(...
    'NumPyramidLevels',3,...
    'MaxBidirectionalError',1,... % inf->1
    'BlockSize',[31,31],...
    'MaxIterations',30); 

initialize(pointTracker,corners.Location,img0);


% Identify points in new frame
[trackedPoints,point_validity] = pointTracker(img1);

% Only Keep valid matches and discard the rest
matchedPoints0 = corners.Location(point_validity,:);
matchedPoints1 = trackedPoints(point_validity,:);

% Construct camera-parameter object for other functions
% Camera intrinsics are define as the transpose of the lecture definition
cameraParams = cameraParameters('IntrinsicMatrix',K');

%{
Compute Fundamental Matrix
F = estimateFundamentalMatrix(matchedPoints0,matchedPoints1,...
    'Method','RANSAC',...
    'DistanceType','Sampson',...
    'DistanceThreshold',0.01,...
    'Confidence',99,...          
    'InlierPercentage',50,...
    'ReportRuntimeError',true);
%}


% Compute essential matrix
[E, epipolarInliers] = estimateEssentialMatrix(matchedPoints0,matchedPoints1,cameraParams,...
    'MaxNumTrials',500, ...
    'Confidence',99.9, ... %99 -> 99.9
    'MaxDistance',0.1);

inlierPoints0 = matchedPoints0(epipolarInliers,:);
inlierPoints1 = matchedPoints1(epipolarInliers,:);

% Compute relative orientation of the camera of img1 to img0
[orient,loc,validPointsFraction] = relativeCameraPose(E,cameraParams,inlierPoints0,inlierPoints1);

% Compute Camera Matrix for img0 (also world frame)
tform0 = rigid3d; % Creates an identity transformation
camMatrix0 = cameraMatrix(cameraParams, tform0);

% Compute Camera Matrix for img1 (also world frame)
cameraPose = rigid3d(orient, loc);
tform1 = cameraPoseToExtrinsics(cameraPose);
camMatrix1 = cameraMatrix(cameraParams, tform1);


% Compute the 3-D points
points3D = triangulate(matchedPoints0, matchedPoints1, camMatrix0, camMatrix1);

%{
% Get the color of each reconstructed point
numPixels = size(img0, 1) * size(img0, 2);
allColors = reshape(img0, [numPixels, 3]);
colorIdx = sub2ind([size(img0, 1), size(img0, 2)], round(matchedPoints0(:,2)), ...
    round(matchedPoints1(:, 1)));
color = allColors(colorIdx, :);
%}

% Create the point cloud
ptCloud = pointCloud(points3D);   %, 'Color', color);


%tform = estimateGeometricTransform3D(corners.Location,matchedPoints,'rigid');

%% Bootstrap: Plot results
%Show used frames
figure(1);
subplot(2,1,1);
imshow(img0);
title('Matched Features with FASTCorners and KLT');
subplot(2,1,2);
imshow(img1);
title('Remaining Inlier Points');



%Show matched features to check performance
figure(2);
subplot(2,1,1);
showMatchedFeatures(img0,img1,matchedPoints0,matchedPoints1);
title('Matched Features with FASTCorners and KLT');
subplot(2,1,2);
showMatchedFeatures(img0,img1,inlierPoints0,inlierPoints1);
title('Remaining Inlier Points');


%{
% Visualize the camera locations and orientations
cameraSize = 0.3;
figure
plotCamera('Size', cameraSize, 'Color', 'r', 'Label', '1', 'Opacity', 0);
hold on
grid on
plotCamera('Location', loc, 'Orientation', orient, 'Size', cameraSize, ...
    'Color', 'b', 'Label', '2', 'Opacity', 0);

% Visualize the point cloud
pcshow(ptCloud, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
    'MarkerSize', 45);

% Rotate and zoom the plot
camorbit(0, -30);
camzoom(1.5);

% Label the axes
xlabel('x-axis');
ylabel('y-axis');
zlabel('z-axis');

title('Up to Scale Reconstruction of the Scene');

%}

%% Some Tests
corners1 = detectFASTFeatures(img1,...
    'MinQuality',0.1,...
    'MinContrast',0.2,...
    'ROI',[1 1 size(img1,2) size(img1,1)]);

corners1 = corners1.Location;
samePointPixelThreshhold = 1;
[Locb,distanceToClosestPoint] = dsearchn(inlierPoints1,corners1);
L = distanceToClosestPoint > samePointPixelThreshhold;


size(corners1,1)
size(inlierPoints1,1)

sum(L)
figure(3)
imshow(img1); hold on;
scatter(corners1(:,1),corners1(:,2));
scatter(inlierPoints1(:,1),inlierPoints1(:,2),'green','+');
scatter(corners1(~L,1),corners1(~L,2),'red','x');
legend('New Candidates', 'KLT Points', 'Discarded')

return;
%% Continuous operation
range = (bootstrap_frames(2)+1):last_frame;
for i = range
    fprintf('\n\nProcessing frame %d\n=====================\n', i);
    if ds == 0
        image = imread([kitti_path '/05/image_0/' sprintf('%06d.png',i)]);
    elseif ds == 1
        image = rgb2gray(imread([malaga_path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(i).name]));
    elseif ds == 2
        image = im2uint8(rgb2gray(imread([parking_path ...
            sprintf('/images/img_%05d.png',i)])));
    else
        assert(false);
    end
    % Makes sure that plots refresh.    
    pause(0.01);
    
    prev_img = image;
end