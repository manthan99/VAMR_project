%% Setup
clear all;
close all;
rng(1);

ds = 0; % 0: KITTI, 1: Malaga, 2: parking

if ds == 0
    % need to set kitti_path to folder containing "05" and "poses"
%     assert(exist('../data/kitti05/kitti', 'var') ~= 0);
    kitti_path = '../data/kitti05/kitti';
    ground_truth = load([kitti_path, '/poses/05.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    last_frame = 4540;
    K = [7.188560000000e+02 0 6.071928000000e+02
        0 7.188560000000e+02 1.852157000000e+02
        0 0 1];
    
elseif ds == 1
    % Path containing the many files of Malaga 7.
%     assert(exist('../data/malaga-urban-dataset-extract-07', 'var') ~= 0);
    malaga_path = '../data/malaga-urban-dataset-extract-07'
    images = dir([malaga_path,
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    left_images = images(3:2:end);
    last_frame = length(left_images);
    K = [621.18428 0 404.0076
        0 621.18428 309.05989
        0 0 1];
elseif ds == 2
    % Path containing images, depths and all...
%     assert(exist('../data/parking', 'var') ~= 0);
    parking_path = '../data/parking'
    last_frame = 598;
    K = load([parking_path, '/K.txt']);
    K
    ground_truth = load([parking_path, '/poses.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
else
    assert(false);
end

%% Bootstrap
% need to set bootstrap_frames
if ds == 0
    img0 = imread([kitti_path '/05/image_0/',sprintf('%06d.png',52)]);
    img1 = imread([kitti_path '/05/image_0/',sprintf('%06d.png',54)]);
    [height, width] = size(img0);
    intrinsics = cameraIntrinsics([K(1,1),K(2,2)],[K(1,3),K(2,3)], [height, width]);
    
% elseif ds == 1
%     img0 = rgb2gray(imread([malaga_path ...
%         '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
%         left_images(bootstrap_frames(1)).name]));
%     img1 = rgb2gray(imread([malaga_path ...
%         '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
%         left_images(bootstrap_frames(2)).name]));
% elseif ds == 2
%     img0 = rgb2gray(imread([parking_path
%         sprintf('/images/img_%05d.png',bootstrap_frames(1))]));
%     img1 = rgb2gray(imread([parking_path ...
%         sprintf('/images/img_%05d.png',bootstrap_frames(2))]));

else
    assert(false);
end   
    corners_0 = detectHarrisFeatures(img0);
    
    %kp_0 = corners_0.selectStrongest(100);
    %corners_1 = detectHarrisFeatures(img1);
    %kp_1 = corners_1.selectStrongest(100);
    %[features_0, valid_corners_0] = extractFeatures(img0, corners_0);
    
    tracker = vision.PointTracker('MaxBidirectionalError',1);
    initialize(tracker,corners_0.Location,img0);
    [corners_1,validity] = tracker(img1);
    matchedPoints0 = corners_0.Location(validity, :);
    matchedPoints1 = corners_1(validity, :);
    
    
    
%     imshow(img1);
%     hold on;
%     plot(matchedPoints1(:, 1), matchedPoints1(:, 2), 'rx', 'Linewidth', 2);
%     matches = linspace(1,size(matchedPoints0,1),335);
%     plotMatches(matches' , matchedPoints1', matchedPoints0');
    
%     matchedPoints0 = flipud(matchedPoints0);
%     matchedPoints1 = flipud(matchedPoints1);
      p1 = matchedPoints0';
      p1 = [p1; ones(1,size(p1,2))];
      p2 = matchedPoints1';
      p2 = [p2; ones(1,size(p2,2))];

    E = estimateEssentialMatrix(p1, p2, K, K);

    % Obtain extrinsic parameters (R,t) from E
    [Rots,u3] = decomposeEssentialMatrix(E);

    % Disambiguate among the four possible configurations
    [R_C2_W,T_C2_W] = disambiguateRelativePose(Rots,u3,p1,p2,K,K);

    % Triangulate a point cloud using the final transformation (R,T)
    M1E = K * eye(3,4);
    M2E = K * [R_C2_W, T_C2_W];
    P = linearTriangulation(p1,p2,M1E,M2E);
    R_C2_W
    T_C2_W
%     scatter3(P(1,:), P(2,:), P(3,:))
    
    F = estimateFundamentalMatrix(matchedPoints0,matchedPoints1,'Method','Norm8Point');
    [R, T, validFraction] = relativeCameraPose(F, intrinsics, ...
        matchedPoints0, matchedPoints1);
    
    [F_ransac, inliersLogicalIndex]   = estimateFundamentalMatrix( ...
    matchedPoints0, matchedPoints1, 'Method','RANSAC',...
    'NumTrials', 1e3);
    
%% Visualize the 3-D scene
figure(1),
subplot(1,3,1)

% R,T should encode the pose of camera 2, such that M1 = [I|0] and M2=[R|t]

% P is a [4xN] matrix containing the triangulated point cloud (in
% homogeneous coordinates), given by the function linearTriangulation
valid_ind = find(P(3,:)>0);
plot3(P(1,valid_ind), P(2,valid_ind), P(3,valid_ind), 'o');

% Display camera pose

plotCoordinateFrame(eye(3),zeros(3,1), 0.8);
text(-0.1,-0.1,-0.1,'Cam 1','fontsize',10,'color','k','FontWeight','bold');

center_cam2_W = -R_C2_W'*T_C2_W;
plotCoordinateFrame(R_C2_W',center_cam2_W, 0.8);
text(center_cam2_W(1)-0.1, center_cam2_W(2)-0.1, center_cam2_W(3)-0.1,'Cam 2','fontsize',10,'color','k','FontWeight','bold');

axis equal
rotate3d on;
grid

% Display matched points
subplot(1,3,2)
imshow(img0,[]);
hold on
plot(p1(1,valid_ind), p1(2,valid_ind), 'ys');
title('Image 1')

subplot(1,3,3)
imshow(img1,[]);
hold on
plot(p2(1,valid_ind), p2(2,valid_ind), 'ys');
title('Image 2')
%%

%     F
%     F_ransac

%     M0 = cameraMatrix(intrinsics,eye(3),[0 0 0]');
%     M1 = cameraMatrix(intrinsics,R,T');
%     
%     [points3d, reprojection_err, valid_ind] = triangulate(matchedPoints0, matchedPoints1, M0, M1);
%     valid_index_3d = find(valid_ind);
%     
% %     scatter3(points3d(valid_index_3d,1), points3d(valid_index_3d,2), points3d(valid_index_3d,3))
%     
%     [R_WC,T_WC, inlier, status] = estimateWorldCameraPose(matchedPoints1(valid_index_3d,:), points3d(valid_index_3d,:), intrinsics, ...
%     'Confidence', 95, 'MaxReprojectionError', 2, 'MaxNumTrials', 1e4);
% 
%     T_WC;
%     R_WC;
%     T_CW = -T_WC
%     R_CW = R_WC'
    
%     matchedPoints0 = matchedPoints0(valid_index_3d,:)
%     out_0 = insertMarker(img0,matchedPoints0,'+');
%     out_1 = insertMarker(img1,matchedPoints1,'+');
    
%     [F_ransac, inliersLogicalIndex]   = estimateFundamentalMatrix( ...
%     matchedPoints0, matchedPoints1, 'Method','RANSAC',...
%     'NumTrials', 1e3);
% 
%     F
%     F_ransac
%     
% %     [F, inliersLogicalIndex] = estimateFundamentalMatrix(matchedPoints0,matchedPoints1,'NumTrials',2000)
% % 
%     inliersIndex  = find(inliersLogicalIndex);
%      [R_r, T_r, validFraction] = relativeCameraPose(F_ransac, intrinsics, ...
%         matchedPoints0(inliersIndex,:), matchedPoints1(inliersIndex,:));
%     
%     M0_r = cameraMatrix(intrinsics,eye(3),[0 0 0]');
%     M1_r = cameraMatrix(intrinsics,R_r,T_r');
% %     
% %     
%     [points3d_r, reprojection_err_r, valid_ind_r] = triangulate(matchedPoints0(inliersIndex,:), matchedPoints1(inliersIndex,:), M0_r, M1_r);
%     valid_index_3d_r = find(valid_ind_r);
% %     [worldOrientation_r,worldLocation_r] = estimateWorldCameraPose(matchedPoints1(valid_index_3d_r,:), points3d_r(valid_index_3d_r,:), intrinsics);
%   [worldOrientation_r,worldLocation_r, inlier, status] = estimateWorldCameraPose(matchedPoints1(valid_index_3d_r,:),  points3d_r(valid_index_3d_r,:), intrinsics, ...
%     'Confidence', 95, 'MaxReprojectionError', 3, 'MaxNumTrials', 1e4);
% 
% %     M2 = cameraMatrix(intrinsics,worldOrientation,worldLocation);
% %     
%     R
%     R_r
%     T
%     T_r
%     worldLocation
%     worldOrientation
%     worldLocation_r
%     worldOrientation_r
%     
%     T_r./worldLocation_r
%     T./worldLocation
% %   
% %     reprojection_err;
% %     
% 
%     scatter3(points3d(:,1), points3d(:,2), points3d(:,3))
%     
%     subplot(1,2,1);
%     imshow(out_0);
%     subplot(1,2,2);
%     imshow(out_1);


%% Continuous operation
prev_img = img0;
inlier_mask = valid_ind;
range = (55):last_frame;
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
%     if i==7
%         continue;
%     end    
    query_image = image;
    
    matchedPoints0 = matchedPoints0(inlier_mask, :);
    reset(tracker);
    release(tracker);
    initialize(tracker,matchedPoints0,prev_img);
    

    
    [corners_1,validity] = tracker(query_image);
    matchedPoints1 = corners_1(validity, :);
    points3d_NEW = P(1:3,validity);
    
    subplot(1,3,2)
    imshow(prev_img,[]);
    hold on
    plot(matchedPoints0(:,1), matchedPoints0(:,2), 'ys');
    title('Image 1')

    subplot(1,3,3)
    imshow(query_image,[]);
    hold on
    plot(matchedPoints1(:,1), matchedPoints1(:,2), 'ys');
    title('Image 2')
    
    plot(matchedPoints1(:, 1), matchedPoints1(:, 2), 'rx', 'Linewidth', 2);
    matches = linspace(1,size(matchedPoints0,1),335);
    plotMatches(matches' , matchedPoints1', matchedPoints0(validity, :)');
    
    matchedPoints1 = fliplr(matchedPoints1);
    [R_C_W, t_C_W, inlier_mask, max_num_inliers_history, num_iteration_history] = ...
    ransacLocalization(matchedPoints1', points3d_NEW, K);

    disp('Found transformation T_C_W = ');
    disp([R_C_W t_C_W; zeros(1, 3) 1]);
    disp('Estimated inlier ratio is');
    disp(nnz(inlier_mask)/numel(inlier_mask));
    
%     plot(max_num_inliers_history);
%     title('Maximum inlier count over RANSAC iterations.');
    
%     [R_WC,T_WC, inlier, status] = estimateWorldCameraPose(matchedPoints1, points3d_NEW, intrinsics, ...
%     'Confidence', 95, 'MaxReprojectionError', 3, 'MaxNumTrials', 1e4);
    
%     inliers = find(inlier);
%     size(inliers)
%     T_CW = -T_WC
%     R_CW = R_WC'
    figure(1),
    subplot(1,3,1)
    center_cam2_W = -R_C_W'*t_C_W;
    subplot(1,3,1)
    plot3(points3d_NEW(1,inlier_mask), points3d_NEW(2,inlier_mask), points3d_NEW(3,inlier_mask), 'o');
    plotCoordinateFrame(R_C_W',center_cam2_W, 0.8);
    text(center_cam2_W(1)-0.1, center_cam2_W(2)-0.1, center_cam2_W(3)-0.1,'Cam 2','fontsize',10,'color','k','FontWeight','bold');
    hold off;
    
    % Makes sure that plots refresh.    
    pause(0.01);
    
    prev_img = query_image;
    P = points3d_NEW;
    matchedPoints0 = fliplr(matchedPoints1);
end
