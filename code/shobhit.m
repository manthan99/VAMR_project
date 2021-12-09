clear all;
close all;
rng(7) %7,8,9
%% Setup
ds = 0; % 0: KITTI, 1: Malaga, 2: parking

if ds == 0
    % need to set kitti_path to folder containing "05" and "poses"
    kitti_path = '../data/kitti05/kitti';
    ground_truth = load([kitti_path, '/poses/05.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    last_frame = 4540;
    K = [7.188560000000e+02 0 6.071928000000e+02
        0 7.188560000000e+02 1.852157000000e+02
        0 0 1];
    
elseif ds == 1
    % Path containing the many files of Malaga 7.
    malaga_path = '../malaga-urban-dataset-extract-07';
    assert(exist('malaga_path', 'var') ~= 0);
    images = dir([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    left_images = images(3:2:end);
    last_frame = length(left_images);
    K = [621.18428 0 404.0076
        0 621.18428 309.05989
        0 0 1];
elseif ds == 2
    % Path containing images, depths and all...
    parking_path = '../parking';
    assert(exist('parking_path', 'var') ~= 0);
    last_frame = 598;
    K = load([parking_path '/K.txt']);
     
    ground_truth = load([parking_path '/poses.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
else
    assert(false);
end

%% Bootstrap
% need to set bootstrap_frames
if ds == 0
    bootstrap_frames = [0,3];
    img0 = imread([kitti_path '/05/image_0/' ...
        sprintf('%06d.png',bootstrap_frames(1))]);
    img1 = imread([kitti_path '/05/image_0/' ...
        sprintf('%06d.png',bootstrap_frames(2))]);
    intrinsics = cameraIntrinsics([K(1,1), K(2,2)], [K(1,3), K(2,3)], size(img0'));
    points0 = detectMinEigenFeatures(img0);
%     points0 = selectStrongest(points0,min(300,floor(length(points0)*0.8)));
    points0 = points0.Location;

    tracker = vision.PointTracker('MaxBidirectionalError',0.1); %0.1
    initialize(tracker,points0,img0);
    [points1, KLT_inliers] = tracker(img1);
    points0 = points0(KLT_inliers,:);
    points1 = points1(KLT_inliers,:);

    [M, MSAC_inliers, status] = estimateEssentialMatrix(points0, points1, intrinsics);
    points0 = points0(MSAC_inliers,:);
    points1 = points1(MSAC_inliers,:);
    
    %in matlab sfm, its recommended to match a dense set of points now
    [R, T] = relativeCameraPose(M, intrinsics, points0, points1);
    tform0 = rigid3d;
    cam_mat_0 = cameraMatrix(intrinsics, eye(3),[0 0 0]');
    cameraPose1 = rigid3d(R,T);
    [R1,T1] = cameraPoseToExtrinsics(R,T);
    cam_mat_1 = cameraMatrix(intrinsics, R1, T1);
%     M_init = cameraMatrix(intrinsics,eye(3),[0 0 0]');
%     M_curr = cameraMatrix(intrinsics,R',-R'*T');

    [worldPoints,reproj_error,validIndex] = triangulate(points0, points1, cam_mat_0, cam_mat_1);
    ptCloud = pointCloud(worldPoints);

    %visualize
    figure (1),
    pointImage0 = insertMarker(img0,points0,'+','Color','green');
    imshow(pointImage0);
    hold on

    pointImage1 = insertMarker(img1,points1,'+','Color','red');
%     R = refinedPose.Rotation;
%     T = refinedPose.Translation;
%     Mm = [R, T'];
    p_reproj = worldToImage(intrinsics,R1,T1,worldPoints);
    
    figure (2),
    imshow(pointImage1);
    hold on;
    plot(p_reproj(:,1), p_reproj(:,2), 'ys');
    hold on

    cameraSize = 0.3;
    figure
    xlim([-100,100])
    ylim([-100,100])
    zlim([-100,100])
    plotCamera('Size', cameraSize, 'Color', 'r', 'Label', '1', 'Opacity', 0);
    hold on
    grid on
    plotCamera('Location', T, 'Orientation', R, 'Size', cameraSize, ...
        'Color', 'b', 'Label', '2', 'Opacity', 0);
    
    % Visualize the point cloud
    pcshow(ptCloud, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
        'MarkerSize', 45);

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

%% Continuous operation
range = (bootstrap_frames(2)+1):last_frame;
Trajectory = zeros(3,4,length(range)+bootstrap_frames(2)+1);
Trajectory(:,:,1) = [eye(3), [0;0;0]];
Trajectory(:,:,bootstrap_frames(2)+1) = [R,T'];
for i = range
    fprintf('\n\nProcessing frame %d\n=====================\n', i);
    if ds == 0
        img0 = img1;
        points0 = points1;
        img1 = imread([kitti_path '/05/image_0/' sprintf('%06d.png',i)]);
        
        tracker = vision.PointTracker('MaxBidirectionalError',0.5); %0.1
        initialize(tracker, points0, img0);
        [points1, KLT_inliers, KLT_scores] = tracker(img1);
        sum(KLT_inliers)/length(KLT_inliers)
%         points0 = imagepoints(KLT_inliers,:);
        points1 = points1(KLT_inliers,:); %2d correspondences in new image
        
        worldPoints = worldPoints(KLT_inliers,:); %leftover 3d points
        ptCloud = pointCloud(worldPoints);

        [R,T, inliersIdx] = estimateWorldCameraPose(points1, worldPoints, intrinsics);
        Trajectory(:,:,i+1) = [R,T'];
        sum(inliersIdx)

        %visualize
        for j=bootstrap_frames(2)+1:i+1
            plotCamera('Location', Trajectory(:,4,j), 'Orientation', Trajectory(:,1:3,j), 'Size', cameraSize, ...
            'Label', num2str(j-1), 'Opacity', 0);
            hold on
        end
        hold on
        grid on
        % Visualize the point cloud
        pcshow(ptCloud, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
            'MarkerSize', 45);
        hold off

    elseif ds == 1
        img1 = rgb2gray(imread([malaga_path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(i).name]));
    elseif ds == 2
        img1 = im2uint8(rgb2gray(imread([parking_path ...
            sprintf('/images/img_%05d.png',i)])));
    else
        assert(false);
    end
    % Makes sure that plots refresh.    
    pause(0.5);    
end