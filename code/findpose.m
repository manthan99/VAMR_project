function [R, T, worldPoints, X_id, outlier_id, points1, points_outliers] = findpose(query_image, ds_vars, prev_state, klt_vars)
% This function finds the camera pose of the current frame
% Inputs: query_image: current frame, ds_vars: dataset variables, prev_state: continuous operation state struct, klt_vars: parameters for klt
% Outputs: R,T - camera pose, worldPoints: 3d landmarks, X_id: 3d landmark global id vector, outlier_id: outlier 3d landmarks gloabl id vector
% points1: inlier imagepoints corresponding to landmarks, points_outliers: outlier imagepoints

    tracker = vision.PointTracker('MaxBidirectionalError',klt_vars.bidir_error, 'BlockSize', klt_vars.block);
    initialize(tracker, prev_state.P, prev_state.prev_img); %initialize klt on previous image and its keypoints
    
    [points1, inliers] = tracker(query_image); %track those features in query_image
    disp(["Current tracked points", sum(inliers)]);
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
    
%   % Motion only BA, nonlinear reproj error optimisation
    curr_pose = rigid3d(R,T);
    refinedPose = bundleAdjustmentMotion(worldPoints(inlierIDX,:),points1(inlierIDX,:),curr_pose, ds_vars.intrinsics);
    R = refinedPose.Rotation;
    T = refinedPose.Translation;
    worldPoints = worldPoints(inlierIDX,:);
    points1 = points1(inlierIDX,:);
    X_id = X_id(inlierIDX);
end