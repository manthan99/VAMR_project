function [R, T, X_id, outlier_id, points1, points_outliers] = findpose(query_image, ds_vars, prev_state)
    
    tracker = vision.PointTracker('MaxBidirectionalError',1);
    initialize(tracker, prev_state.P, prev_state.prev_img); %initialize klt on previous image and its keypoints
    
    [points1, inliers] = tracker(query_image); %track those features in query_image
    disp(["Current tracked points", sum(inliers)]);
%     points0 = prev_state.P(inliers,:); %prev_img keypoints that are tracked in query_image
    points1 = points1(inliers,:); %keypoints in query_image that are tracked
    worldPoints = prev_state.X(inliers,:); %their corresponding worldPoints that we know from before
    X_id = prev_state.X_id(inliers);

    % If using Estimate World Camera Pose, 2D-3D to estimate pose
    [R,T, inlierIDX] = estimateWorldCameraPose(...
        points1, worldPoints, ds_vars.intrinsics, 'Confidence', 95, 'MaxReprojectionError', 3, 'MaxNumTrials', 1000);
    outlierIDX = ~inlierIDX;
    inlierIDX = find(inlierIDX);
    outlierIDX = find(outlierIDX);
    points_outliers = points1(outlierIDX,:);
    outlier_id = X_id(outlierIDX);
    % If P3p RANSAC
%     [R,T, inlierIDX] = ransacLocalization(points1', worldPoints', K);
%     [R,T] = extrinsicsToCameraPose(R,T);
%     inlierIDX = inlierIDX';
%     inlierIDX = find(inlierIDX);
    
%   % Motion only BA, nonlinear reproj error optimisation
    curr_pose = rigid3d(R,T);
    refinedPose = bundleAdjustmentMotion(worldPoints(inlierIDX,:),points1(inlierIDX,:),curr_pose, ds_vars.intrinsics);
    R = refinedPose.Rotation;
    T = refinedPose.Translation;
%     worldPoints = worldPoints(inlierIDX,:);
    points1 = points1(inlierIDX,:);
    X_id = X_id(inlierIDX);
end