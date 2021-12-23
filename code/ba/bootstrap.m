function [inlierCurrPts, worldPoints, R1, T1, R, T, currImg, ds_vars, i] = bootstrap(ds_vars, harris_vars, klt_vars)
    i = 1;
    
    img0 = load_image(ds_vars, i);
    
    ds_vars.intrinsics = cameraIntrinsics([ds_vars.K(1,1),ds_vars.K(2,2)],[ds_vars.K(1,3),ds_vars.K(2,3)], size(img0));

    corners0 = detect_features(harris_vars, img0);

    figure(3),
    imshow(img0, []);
    hold on
    plot(corners0(:,1),corners0(:,2), 'gs');
    hold on
    
    tracker = vision.PointTracker('MaxBidirectionalError',klt_vars.bidir_error, 'BlockSize', klt_vars.block);
    initialize(tracker,corners0,img0);
    
    isBootStrapped  = false;
    i = i+1;
    
    %Loop until a proper initialization frame is found
    while(~isBootStrapped && i<ds_vars.last_frame)
        currImg = load_image(ds_vars, i);
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