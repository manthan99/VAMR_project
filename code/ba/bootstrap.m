function [inlierCurrPts, worldPoints, R1, T1, currImg, ds_vars, i] = bootstrap(ds_vars, harris_vars)
    i = 1;
    
    if ds_vars.ds == 0
        img0 = imread([ds_vars.path '/05/image_0/',sprintf('%06d.png',i)]);
    end
    if ds_vars.ds == 1
        img0 = rgb2gray(imread([ds_vars.path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            ds_vars.left_images(i).name]));
    end
    if ds_vars.ds == 2
        img0 = im2uint8(rgb2gray(imread([ds_vars.path ...
            sprintf('/images/img_%05d.png',i)])));
    end
    
    ds_vars.intrinsics = cameraIntrinsics([ds_vars.K(1,1),ds_vars.K(2,2)],[ds_vars.K(1,3),ds_vars.K(2,3)], size(img0'));

    query_harris = harris(img0, harris_vars.harris_patch_size, harris_vars.harris_kappa);
    corners0 = selectKeypoints(...
    query_harris, harris_vars.num_keypoints, harris_vars.nonmaximum_supression_radius);
    corners0 = flipud(corners0)';

    figure(3),
    imshow(img0, []);
    hold on
    plot(corners0(:,1),corners0(:,2), 'gs');
    hold on
    
    tracker = vision.PointTracker('MaxBidirectionalError',1);
    initialize(tracker,corners0,img0);
    
    isBootStrapped  = false;
    i = i+1;
    
    %Loop until a proper initialization frame is found
    while(~isBootStrapped && i<ds_vars.last_frame)
        if ds_vars.ds == 0
            currImg = imread([ds_vars.path '/05/image_0/',sprintf('%06d.png',i)]);
        end
        if ds_vars.ds == 1
        currImg = rgb2gray(imread([ds_vars.path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            ds_vars.left_images(i).name]));
        end
        if ds_vars.ds == 2
            currImg = im2uint8(rgb2gray(imread([ds_vars.path ...
            sprintf('/images/img_%05d.png',i)])));
        end
        disp(['Current Frame is ', num2str(i)]);
        
        [corners1,inliers] = tracker(currImg);
        points0 = corners0(inliers,:);
        points1 = corners1(inliers,:);

        minMatches = 150;
        i=i+1;
        
        %If the min number of matches are less than the threshold, skip
        %this frame
        if size(inliers, 1) < minMatches
            disp(['Doesnt satisfy minMatches threshold']);
            continue
        end
        
        [F, inliersLogicalIndex]   = estimateFundamentalMatrix( ...
        points0, points1, 'Method','RANSAC',...
        'NumTrials', 1e3, 'DistanceThreshold', 0.1);
        
        inliersIndex  = find(inliersLogicalIndex);
        inlierInitPts  = points0(inliersIndex,:);
        inlierCurrPts = points1(inliersIndex,:);
        
        figure(1),
        showMatchedFeatures(img0,currImg,inlierInitPts,inlierCurrPts);
        legend('matched points 1','matched points 2');
        
        [R, T, validFraction] = relativeCameraPose(F, ds_vars.intrinsics, ...
        inlierInitPts, inlierCurrPts);
        
        % If not enough inliers are found, move to the next frame
        if validFraction < 0.90 || numel(size(R))==3
            disp('Doesnt satisfy valid fraction threshold');
            continue
        end
        
        % Triangulate two views to obtain 3-D map points
        minParallax = 0.5; % In degrees
        cam_mat_0 = cameraMatrix(ds_vars.intrinsics, eye(3),[0 0 0]');
        [R1,T1] = cameraPoseToExtrinsics(R,T);
        cam_mat_1 = cameraMatrix(ds_vars.intrinsics, R1, T1);
        
        [worldPoints,reprojectionErrors,isInFront] = triangulate(inlierInitPts, inlierCurrPts,cam_mat_0,cam_mat_1);

        % Filter points by view direction and reprojection error
        minReprojError = 2;
        inlierIdx  = isInFront & reprojectionErrors < minReprojError;
        worldPoints  = worldPoints(inlierIdx ,:);
        inlierCurrPts = inlierCurrPts(inlierIdx ,:);
        inlierInitPts = inlierInitPts(inlierIdx ,:);
       
        % A good two-view with significant parallax
        ray1       = worldPoints - [0 0 0];
        ray2       = worldPoints - T;
        cosAngle   = sum(ray1 .* ray2, 2) ./ (vecnorm(ray1, 2, 2) .* vecnorm(ray2, 2, 2));

        % Check parallaxsize(R)
        prlx_ind = find(cosAngle < cosd(minParallax) & cosAngle>0);
        
      
        if size(prlx_ind,1) < 150 || size(prlx_ind,1)/size(worldPoints,1) < 0.4
            disp(['Doesnt satisfy parallax threshold']);
            continue
        end
        
        prlx_ind = find(cosAngle < cosd(0.3) & cosAngle>0);
        
        worldPoints = worldPoints(prlx_ind,:); %3d points
        inlierCurrPts = inlierCurrPts(prlx_ind,:); %points in current frame
        inlierInitPts = inlierInitPts(prlx_ind,:); %points in 0th frame
        
        p_reproj = worldToImage(ds_vars.intrinsics,R1,T1,worldPoints);
        
        figure(5),
        imshow(currImg, []);
        hold on
        plot(p_reproj(:,1), p_reproj(:,2), 'ys');
        hold on
        plot(inlierCurrPts(:,1), inlierCurrPts(:,2), 'rx');
        hold on
        
        isBootStrapped  = true;
        disp(['Map initialized with frame 0 and frame ', num2str(i-1)]) %(i-1)th image is current image
    end 
end