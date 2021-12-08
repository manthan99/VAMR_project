%% Setup
clear all;
close all;

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
    
    corners0 = detectHarrisFeatures(img0);
    corners0 = corners0.Location;
    tracker = vision.PointTracker('MaxBidirectionalError',1);
    initialize(tracker,corners0,img0);
    
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
        
        [corners1,inliers] = tracker(currImg);
        points0 = corners0(inliers,:);
        points1 = corners1(inliers,:);

        minMatches = 100;
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
        
        [R, T, validFraction] = relativeCameraPose(F, intrinsics, ...
        inlierInitPts, inlierCurrPts);
        
        % If not enough inliers are found, move to the next frame
        if validFraction < 0.90 || numel(size(R))==3
            disp(['Doesnt satisfy valid fraction threshold']);
            continue
        end
        
        % Triangulate two views to obtain 3-D map points
        minParallax = 0.5; % In degrees
        cam_mat_0 = cameraMatrix(intrinsics, eye(3),[0 0 0]');
        [R1,T1] = cameraPoseToExtrinsics(R,T);
        cam_mat_1 = cameraMatrix(intrinsics, R1, T1);
        
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

        % Check parallax
        prlx_ind = find(cosAngle < cosd(minParallax) & cosAngle>0);
        
      
        if size(prlx_ind,1) < 150 || size(prlx_ind,1)/size(worldPoints,1) < 0.4
            disp(['Doesnt satisfy parallax threshold']);
            continue
        end
        
        prlx_ind = find(cosAngle < cosd(0.3) & cosAngle>0);
        
        worldPoints = worldPoints(prlx_ind,:);
        inlierCurrPts = inlierCurrPts(prlx_ind,:);
        inlierInitPts = inlierInitPts(prlx_ind ,:);
        
        p_reproj = worldToImage(intrinsics,R1,T1,worldPoints);
        
        figure(5),
        imshow(currImg, []);
        hold on
        plot(p_reproj(:,1), p_reproj(:,2), 'ys');
        hold on
        plot(inlierCurrPts(:,1), inlierCurrPts(:,2), 'rx');
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
 plotCamera('Size',1,'Orientation',R,'Location',...
     T);
 hold on
 plotCamera('Size',1,'Orientation',eye(3),'Location',...
     [0 0 0]');
 hold on


%% Continuous Operation Setup

%P X C F T setup for teh first frame

prev_img = currImg;
P_prev = inlierCurrPts;
X_prev = worldPoints;

%Detect Harris Features
corners0 =  detectHarrisFeatures(prev_img);
corners0 = corners0.Location;

%Remove Duplicates from corner0 corresponding to P_prev (Note more
%duplicates than actual are removed due to tolerance issues)
[L,Locb] = ismembertol(corners0, P_prev, 0.008,'ByRows',true);
L = ~L;
corners0 = corners0(L,:);

% figure(3),
% imshow(prev_img, []);
% hold on
% plot(corners0(:,1),corners0(:,2), 'gs');
% hold on
% plot(P_prev(:,1), P_prev(:,2), 'ys');
% hold on

%Build the C Matrix and F Matrix is also same for the first frame
C_prev = corners0;
F_prev = corners0;

%Convert Affine matrix to a single row vector
%Note, here R and T are transformation from world frame to camera frame so
%can be directly used to for the cameraMatrix function before triangulation
A_prev = [R1,T1'];
A_prev = reshape(A_prev,1,[]);
A_prev = repmat(A_prev,size(C_prev,1),1);

prev_state = struct;
prev_state.P = P_prev;
prev_state.X = X_prev;
prev_state.C = C_prev;
prev_state.F = F_prev;
prev_state.A = A_prev;

%% Continuous operation

range = i:last_frame;
Trajectory = zeros(3,4,length(range)+1);
Trajectory(:,:,1) = [R1, T1'];
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
    
    query_image = image;
    tracker = vision.PointTracker('MaxBidirectionalError',1);
    initialize(tracker,prev_state.P,prev_img);
    
    [points1, inliers] = tracker(query_image);
    points0 = prev_state.P(inliers,:);
    points1 = points1(inliers,:);
    worldPoints = prev_state.X(inliers,:);
    
    figure(3),
    showMatchedFeatures(prev_img,query_image,points0,points1, 'Montage');
    legend('matched points 1','matched points 2');

    [R,T, inlierIDX] = estimateWorldCameraPose...
        (points1,worldPoints,intrinsics, 'Confidence', 95, 'MaxReprojectionError', 4, 'MaxNumTrials', 2e3);
    inlierIDX = find(inlierIDX);
    
%   % Motion only BA
    curr_pose = rigid3d(R,T);
    refinedPose = bundleAdjustmentMotion(worldPoints(inlierIDX,:),points1(inlierIDX,:),curr_pose,intrinsics);
    R = refinedPose.Rotation;
    T = refinedPose.Translation;

    [R1,T1] = cameraPoseToExtrinsics(R,T);
   
    p_reproj = worldToImage(intrinsics,R1,T1,worldPoints);
    figure(6),
    imshow(query_image, []);
    hold on
    plot(p_reproj(:,1), p_reproj(:,2), 'ys');
    hold on
    plot(points1(:,1), points1(:,2), 'rx');
    hold on
    
%     figure(2),
    R_viz = R;
    T_viz = T;
    Trajectory(:,:,i-range(1)+2) = [R,T'];

%     %Now do updates for the state
    C_new = detectHarrisFeatures(query_image);
    C_new = C_new.Location;
    
%     figure(7),
%     imshow(query_image, []);
%     hold on
%     plot(C_new(:,1),C_new(:,2), 'rx');
%     hold on
    
    L = ismembertol(C_new, points1, 0.008,'ByRows',true);
    L = ~L;
    C_new = C_new(L,:);
    
%     figure(7),
% %     imshow(query_image, []);
% %     hold on
%     plot(points1(:,1),points1(:,2), 'gs');
%     hold on
%     plot(C_new(:,1), C_new(:,2), 'ys');
%     hold on
    
    tracker = vision.PointTracker('MaxBidirectionalError',1);
    initialize(tracker,prev_state.C,prev_img);
    
    [C_matched, inliers] = tracker(query_image);
    
    C_matched = C_matched(inliers,:);
    C_old = prev_state.C(inliers,:);
    F_old = prev_state.F(inliers,:);
    A_old = prev_state.A(inliers,:);
    
    [L,Locb] = ismembertol(C_new, C_matched, 0.008,'ByRows',true);
    F_new = C_new;
    A_new = [R1,T1'];
    A_new = reshape(A_new,1,[]);
    A_new = repmat(A_new,size(C_new,1),1);
    
    F_new(L,:) = F_old(Locb(L),:);
    A_new(L,:) = A_old(Locb(L),:);
    
    %Need to remove the duplicates of F
    [F_new,ia,ic] = unique(F_new,'rows','stable');
    A_new = A_new(ia,:);
    C_new = C_new(ia,:);
    
    %Sort according to get same Transformations together
    [A_new,index] = sortrows(A_new);
    C_new = C_new(index,:);
    F_new = F_new(index,:);
    
    X_add = [];
    P_add = [];
    rem_idx = [];
        
    %Now check for new triangulations
    if size(worldPoints,1) < 150
        
        %Get blocks of F with same A
        [A_mat,ia,ic] = unique(A_new,'rows','stable');
        
    
        for i=1:size(ia,1)
            start_ind = ia(i);

            if i < size(ia,1)
                end_ind = ia(i+1);
            else
                end_ind = size(A_new,1);
            end

            % Triangulate two views to obtain 3-D map points
            minParallax = 1; % In degrees
            A = reshape(A_mat(i,:),3,4);
            R = A(1:3,1:3);
            T = A(1:3,4)';
            cam_mat_0 = cameraMatrix(intrinsics, R, T);
            cam_mat_1 = cameraMatrix(intrinsics, R1, T1);

            if(T == T1 | end_ind - start_ind < 10)
                %Skip the iteration since they are the current points
                continue
            end

            [XYZpts,reprojectionErrors,isInFront] = triangulate(F_new(start_ind:end_ind,:), C_new(start_ind:end_ind,:),cam_mat_0,cam_mat_1);

            % Filter points by view direction and reprojection error
            minReprojError = 2;
            inlierIdx  = isInFront & reprojectionErrors < minReprojError;
            XYZpts  = XYZpts(inlierIdx ,:);
            Cpts = C_new(start_ind:end_ind,:);
            Cpts = Cpts(inlierIdx ,:);
            rem_ind = start_ind:end_ind;
            rem_ind = rem_ind(inlierIdx);
            % A good two-view with significant parallax
            ray1       = XYZpts - T1;
            ray2       = XYZpts - T;
            cosAngle   = sum(ray1 .* ray2, 2) ./ (vecnorm(ray1, 2, 2) .* vecnorm(ray2, 2, 2));

            % Check parallax
            prlx_ind = find(cosAngle < cosd(minParallax) & cosAngle>0);


            %Update the values of state
            %maintain P_add, X_add Matrixes and append the new entries to them
            %Maintain a index array which specifies the points to be removed
            %from A, C and F

            X_add = [X_add; XYZpts(prlx_ind,:)];
            P_add = [P_add; Cpts(prlx_ind,:)];
            rem_idx = [rem_idx, rem_ind(prlx_ind)];

        end
    
    [C_new,PS] = removerows(C_new,'ind',rem_idx);
    [F_new,PS] = removerows(F_new,'ind',rem_idx);
    [A_new,PS] = removerows(A_new,'ind',rem_idx);
    end
    
    prev_img = query_image;
    
    prev_state.A = A_new;
    prev_state.C = C_new;
    prev_state.F = F_new;
    prev_state.X = [worldPoints; X_add];
    prev_state.P = [points1; P_add];
    
    figure(2),
    pcshow(prev_state.X,'VerticalAxis','Y','VerticalAxisDir','down', ...
    'MarkerSize',100);
    
    for j=1:i-range(1)+2
        plotCamera('Size',1,'Orientation',Trajectory(:,1:3,j),'Location',Trajectory(:,4,j));
        hold on
    end
    
    close(6);
end
