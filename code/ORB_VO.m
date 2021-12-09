%% Setup
clear all;
close all;

ds = 2; % 0: KITTI, 1: Malaga, 2: parking

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
    malaga_path = '../data/malaga-urban-dataset-extract-07';
    images = dir([malaga_path,'/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    left_images = images(3:2:end);
    last_frame = length(left_images);
    K = [621.18428 0 404.0076
        0 621.18428 309.05989
        0 0 1];
elseif ds == 2
    % Path containing images, depths and all...
    parking_path = '../data/parking';
    last_frame = 598;
    K = load([parking_path, '/K.txt']);
    ground_truth = load([parking_path, '/poses.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
else
    assert(false);
end

%% Bootstrap
% need to set bootstrap_frames
rng(1);

if ds == 0
    img0 = imread([kitti_path '/05/image_0/',sprintf('%06d.png',0)]);
end
if ds == 1
    img0 = rgb2gray(imread([malaga_path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(1).name]));
end
if ds == 2
    img0 = im2uint8(rgb2gray(imread([parking_path ...
            sprintf('/images/img_%05d.png',0)])));
end
    intrinsics = cameraIntrinsics([K(1,1),K(2,2)],[K(1,3),K(2,3)], size(img0'));
    scaleFactor = 1.2;
    numLevels   = 8;
    numPoints   = 2000; %Higher for KITTI Dataset high resolution images
    
    [init_features, init_pts] = Detect_ExtractORB(img0, scaleFactor, ...
        numLevels, numPoints);
 
    isBootStrapped  = false;
    i = 1;
    
    %Loop until a proper initialization frame is found
    while(~isBootStrapped && i<last_frame)
        if ds == 0
            currImg = imread([kitti_path '/05/image_0/',sprintf('%06d.png',i)]);
        end
        if ds == 1
        currImg = rgb2gray(imread([malaga_path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(i).name]));
        end
        if ds == 2
            currImg = im2uint8(rgb2gray(imread([parking_path ...
            sprintf('/images/img_%05d.png',i)])));
        end
     
        
        disp(['Current Frame is ', num2str(i)]);
        
        [curr_features, curr_pts] = Detect_ExtractORB(currImg, scaleFactor, ...
        numLevels, numPoints);
%         indexPairs = matchFeatures(init_features.Features, curr_features.Features, 'Unique', true, ...
%         'MaxRatio', 0.8, 'MatchThreshold', 40);
        indexPairs = matchFeatures(init_features, curr_features, 'Unique', true, ...
        'MaxRatio', 0.8, 'MatchThreshold', 40);
        initMatchedPts  = init_pts(indexPairs(:,1),:);
        currMatchedPts = curr_pts(indexPairs(:,2),:);
    
        minMatches = 100;
        i=i+1;
        
        %If the min number of matches are less than the threshold, skip
        %this frame
        if size(indexPairs, 1) < minMatches
            disp(['Doesnt satisfy minMatches threshold']);
            continue
        end
        
        [F, inliersLogicalIndex]   = estimateFundamentalMatrix( ...
        initMatchedPts, currMatchedPts, 'Method','RANSAC',...
        'NumTrials', 1e3, 'DistanceThreshold', 0.1);
        
        inliersIndex  = find(inliersLogicalIndex);
        inlierInitPts  = initMatchedPts(inliersIndex,:);
        inlierCurrPts = currMatchedPts(inliersIndex,:);
        
        figure(1),
        showMatchedFeatures(img0,currImg,inlierInitPts,inlierCurrPts);
        legend('matched points 1','matched points 2');
        
        [relR, relT, validFraction] = relativeCameraPose(F, intrinsics, ...
        inlierInitPts, inlierCurrPts);
        
        % If not enough inliers are found, move to the next frame
        if validFraction < 0.85 || numel(size(relR))==3
            disp(['Doesnt satisfy valid fraction threshold']);
            continue
        end
        
        % Triangulate two views to obtain 3-D map points
        minParallax = 1; % In degrees
        cam_mat_0 = cameraMatrix(intrinsics, eye(3),[0 0 0]');
        [R1,T1] = cameraPoseToExtrinsics(relR,relT);
        cam_mat_1 = cameraMatrix(intrinsics, R1, T1);
    
        M_init = cameraMatrix(intrinsics,eye(3),[0 0 0]');
        M_curr = cameraMatrix(intrinsics,relR',-relR'*relT');
        
        [worldPoints,reprojectionErrors,isInFront] = triangulate(inlierInitPts, inlierCurrPts,cam_mat_0,cam_mat_1);

        % Filter points by view direction and reprojection error
        minReprojError = 2;
        inlierIdx  = isInFront & reprojectionErrors < minReprojError;
        worldPoints  = worldPoints(inlierIdx ,:);
        inlierCurrPts = inlierCurrPts(inlierIdx ,:);
        inlierInitPts = inlierInitPts(inlierIdx ,:);
%         if size(worldPoints,1) < 100
%             disp(['Doesnt satisfy min points with reprojection error threshold']);
%             continue
%         end
       
        % A good two-view with significant parallax
        ray1       = worldPoints - [0 0 0];
        ray2       = worldPoints - relT;
        cosAngle   = sum(ray1 .* ray2, 2) ./ (vecnorm(ray1, 2, 2) .* vecnorm(ray2, 2, 2));

        % Check parallax
        prlx_ind = find(cosAngle < cosd(minParallax) & cosAngle>0);
        
      
        if size(prlx_ind,1) < 80 || size(prlx_ind,1)/size(worldPoints,1) < 0.5
            disp(['Doesnt satisfy parallax threshold']);
            continue
        end
        
        worldPoints = worldPoints(prlx_ind,:);
        inlierCurrPts = inlierCurrPts(prlx_ind,:);
        inlierInitPts = inlierInitPts(prlx_ind ,:);
        
        p_reproj = worldToImage(intrinsics,R1,T1,worldPoints);
        
        figure(5),
        imshow(currImg, []);
        hold on
        plot(p_reproj(:,1), p_reproj(:,2), 'ys');
        hold on
        plot(inlierCurrPts.Location(:,1), inlierCurrPts.Location(:,2), 'rx');
        hold on
        
        isBootStrapped  = true;
        disp(['Map initialized with frame 0 and frame ', num2str(i-1)])
    
end 
%% Visualize the 3-D scene

figure(2),


% R,T should encode the pose of camera 2, such that M1 = [I|0] and M2=[R|t]

 pcshow(worldPoints,'VerticalAxis','Y','VerticalAxisDir','down', ...
     'MarkerSize',30);
 hold on
 plotCamera('Size',1,'Orientation',relR,'Location',...
     relT);
 hold on
 plotCamera('Size',1,'Orientation',eye(3),'Location',...
     [0 0 0]');
 hold on


%% Intermediate Operation

%After Initialization for the next frame find R|T using available points
%Using frame i-2 and frame i+1 recompute matches and triangulate new 3D
%points(which satisfy parallax) which will then be the starting point of
%the continuous operation

%% Continuous operation

prev_img = currImg;
worldOrientation = R1;
worldLocation = T1;

% [prev_features, prev_pts] = extractFeatures(prev_img, inlierCurrPts);
range = i:last_frame;
% inlierCurrPts = inlierCurrPts.Location;
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
    
    [prev_features, prev_pts] = extractFeatures(prev_img, inlierCurrPts);
    query_image = image;
    
    [L,Locb] = ismember(prev_pts.Location, inlierCurrPts.Location, 'rows');
    
    points0 =  inlierCurrPts(Locb,:);
    worldPoints = worldPoints(Locb,:);
    
    [curr_features, curr_pts] = Detect_ExtractORB(query_image, scaleFactor, ...
        numLevels, numPoints);
    
    indexPairs = matchFeatures(prev_features, curr_features, 'Unique', true, ...
        'MaxRatio', 0.8, 'MatchThreshold', 40);
    prevMatchedPts  = prev_pts(indexPairs(:,1),:);
    currMatchedPts = curr_pts(indexPairs(:,2),:);

    worldPoints = worldPoints(indexPairs(:,1),:);
    
    figure(3),
    showMatchedFeatures(prev_img,query_image,prevMatchedPts,currMatchedPts, 'Montage');
    legend('matched points 1','matched points 2');

    [worldOrientation,worldLocation, inlierIDX] = estimateWorldCameraPose...
        (currMatchedPts.Location,worldPoints,intrinsics, 'Confidence', 95, 'MaxReprojectionError', 2, 'MaxNumTrials', 1e4);
    inlierIDX = find(inlierIDX);
    
    % Motion only BA
    curr_pose = rigid3d(worldOrientation,worldLocation)
    refinedPose = bundleAdjustmentMotion(worldPoints(inlierIDX,:),currMatchedPts.Location(inlierIDX,:),curr_pose,intrinsics)
    worldOrientation = refinedPose.Rotation;
    worldLocation = refinedPose.Translation;
    [R1,T1] = cameraPoseToExtrinsics(worldOrientation,worldLocation);
   
    p_reproj = worldToImage(intrinsics,R1,T1,worldPoints);
    figure(6),
    imshow(query_image, []);
    hold on
    plot(p_reproj(:,1), p_reproj(:,2), 'ys');
    hold on
    plot(currMatchedPts.Location(:,1), currMatchedPts.Location(:,2), 'rx');
    hold on
    
    figure(2),
    plotCamera('Size',1,'Orientation',worldOrientation,'Location',...
     worldLocation);
    hold on
    inlierCurrPts = currMatchedPts;
    prev_img = query_image;
    
end
