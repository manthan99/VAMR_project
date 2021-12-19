%% Setup
clear all;
close all;


ds = 2; % 0: KITTI, 1: Malaga, 2: parking
ba_n = 20; 
harris_vars = struct;
harris_vars.harris_patch_size = 9;
harris_vars.harris_kappa = 0.08;
harris_vars.nonmaximum_supression_radius = 9;
harris_vars.descriptor_radius = 9;
harris_vars.match_lambda = 5;
harris_vars.num_keypoints = 1000;

ds_vars = get_ds_vars(ds);


%% Bootstrap
% need to set bootstrap_frames
rng(1);
    
[inlierCurrPts, worldPoints, R1, T1, currImg, ds_vars, i] = bootstrap(ds_vars, harris_vars);

%% Continuous Operation Setup

%P X C F T setup for the first frame

%for BA function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X_ba = worldPoints; %X_ba will store all 3d points ever found
% P_ba columns - X_id, view_id, x, y
P_ba = [linspace(1,length(worldPoints),length(worldPoints))', ones(length(worldPoints),1)*(i-1), inlierCurrPts];

ViewId = uint32(i-1); 
Orientation = mat2cell(R1, 3);
Location = mat2cell(T1, 1);
pose_table_ba = table(ViewId, Orientation, Location);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

prev_img = currImg;
P_prev = inlierCurrPts;
X_prev = worldPoints;
X_id = (1:length(X_prev))';

query_harris = harris(prev_img, harris_vars.harris_patch_size, harris_vars.harris_kappa);
corners0 = selectKeypoints(...
    query_harris, harris_vars.num_keypoints, harris_vars.nonmaximum_supression_radius);
corners0 = flipud(corners0)';

D_prev = corners0;
E_prev = corners0;
%Remove Duplicates from corner0 corresponding to P_prev (Note more
%duplicates than actual are removed due to tolerance issues)
[L,Locb] = ismembertol(corners0, P_prev, 0.008,'ByRows',true);
L = ~L;
corners0 = corners0(L,:);

%Build the C Matrix and F Matrix is also same for the first frame
C_prev = corners0; %candidates not triangulated
F_prev = corners0; %where candidates were first found

%Convert Affine matrix to a single row vector
%Note, here R and T are transformation from world frame to camera frame so
%can be directly used to for the cameraMatrix function before triangulation
A_prev = [R1,T1'];
A_prev = reshape(A_prev,1,[]);
A_prev = repmat(A_prev,size(C_prev,1),1);

To_prev = [R1,T1']; %poses corresponding to F
To_prev = reshape(To_prev,1,[]);
To_prev = repmat(To_prev,size(D_prev,1),1);

prev_state = struct;
prev_state.prev_img = prev_img;
prev_state.P = P_prev; %2d points that are tracked
prev_state.X = X_prev; %worldPoints that are tracked
prev_state.X_id = X_id; %global id of world points
prev_state.C = C_prev; %candidates not triangulated
prev_state.F = F_prev; %where candidates were first found
prev_state.A = A_prev; %poses corresponding to F
prev_state.D = D_prev; %candidates including the ones that are tracked
prev_state.E = E_prev; %analogous to F for D
prev_state.To = To_prev; %analogous to A for D
prev_state.X_ba = X_ba;
prev_state.P_ba = P_ba;
prev_state.pose_table_ba = pose_table_ba;
prev_state.n_landmark = [length(prev_state.X(:,1))];
% prev_state.Trajectory = zeros(3,4,1);
% prev_state.Trajectory(:,:,1) = [R1', -R1'*T1'];

%% Continuous operation

range = i:ds_vars.last_frame;
% Trajectory = zeros(3,4,length(range)+1);
% Trajectory(:,:,1) = [R1, T1'];
error1_count = 0;
for i = range
    fprintf('\n\nProcessing frame %d\n=====================\n', i);
    
    query_image = load_image(ds_vars, i);
    prev_state.P_ba = prev_state.P_ba(find(prev_state.P_ba(:,2)>i-ba_n), :); %deleting old views 2d pts

    %find pose
    [R, T, X_id, outlier_id, points1, points_outliers] = findpose(query_image, ds_vars, prev_state);
    R_viz = R; %current camera pose
    T_viz = T;

    % Bundle adjustment %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     [L,locb] = ismembertol(prev_state.X_ba, worldPoints, 0.004, 'ByRows',true);
%     if length(find(L==1))>length(worldPoints)
%         disp('error 1');
%         error1_count = error1_count+1;
% %         break
%     end
%     prev_state.P_ba = [prev_state.P_ba; [find(L==1), ones(length(locb(L)),1)*i, points1(locb(L),:)]];
    prev_state.P_ba = [prev_state.P_ba; [X_id, ones(length(X_id),1)*i, points1]];
    prev_state.P_ba = sortrows(prev_state.P_ba);
    Orientation = mat2cell(R,3);
    Location = mat2cell(T,1);
    ViewId = uint32(i);
    prev_state.pose_table_ba = [prev_state.pose_table_ba; table(ViewId, Orientation, Location)];

    x_id = (1:prev_state.P_ba(end,1))'; %unique(prev_state.P_ba(:,1)); vector of all x ids
    hist_xid = histc(prev_state.P_ba(:,1),x_id);
    x_id = x_id(1<hist_xid); %those x_ids which have more than 1 2d points correspondences
    x_ba = prev_state.X_ba(x_id,:);
    point_tracks = [];
    pk=1;
    for j=x_id'
        pts = prev_state.P_ba(find(prev_state.P_ba(:,1)==j), 2:4);  %2d points corresponding to jth 3d point
%         pts = prev_state.P_ba(curr_pointer:curr_pointer+hist_xid(j)-1, 2:4);
        if pk==1
            point_tracks = [point_tracks; pointTrack(pts(:,1), pts(:,2:3))];
            point_tracks = repmat(point_tracks, length(x_id), 1);
        else
            point_tracks(pk) = [pointTrack(pts(:,1), pts(:,2:3))];
        end
        pk = pk+1;
    end
%     [w, locw] = ismembertol(worldPoints, x_ba, 0.008, 'ByRows',true);
%     L=L(x_id); %will give us 1 at those locations of refinedpoints or x_ba that is in worldpoints
%     locb = locb(x_id);
    [xyzRefinedPoints,refinedPoses, reproj_errors] = bundleAdjustment(x_ba, point_tracks, prev_state.pose_table_ba, ds_vars.intrinsics); %length of 3d points and pointtrack array should be same, right?
    prev_state.X_ba(x_id,:) = xyzRefinedPoints;
    prev_state.pose_table_ba = refinedPoses;
%     worldPoints(w,:) = xyzRefinedPoints(locw(w),:);
    worldPoints = prev_state.X_ba(X_id,:);
%     worldPoints(locb(L),:) = xyzRefinedPoints(L,:);
    R = cell2mat(prev_state.pose_table_ba.Orientation(end));
    T = cell2mat(prev_state.pose_table_ba.Location(end));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [R1,T1] = cameraPoseToExtrinsics(R,T);
    
    [A_new, C_new, F_new, To_new, D_new, E_new, prev_state] = update_state(query_image, points1, R1, T1, harris_vars, prev_state);
    
    if i==56
        disp('');
    end
    [X_add, P_add, X_add_id, D_search, E_search, To_search, A_new, C_new, F_new, D_new, E_new, To_new] = triangulate_new(A_new, C_new, F_new, To_new, D_new, E_new, prev_state, worldPoints, points_outliers, outlier_id, ds_vars, R1, T1);
    if length(X_add_id)~=size(X_add,1)
        error1_count = error1_count+1;
        len = min(length(X_add_id),size(X_add,1));
        X_add_id = X_add_id(1:len);
        X_add = X_add(1:len,:);
    end
    %update ba vars
    if ~isempty(X_add)
        [L,locb] = ismember(X_add_id, (1:length(prev_state.X_ba))', "rows");
        notL=~L;
        prev_state.X_ba = [prev_state.X_ba; X_add(notL,:)]; %add only those 3d points that are not already there
%         x_id = length(prev_state.X_ba) - length(find(notL==1))+1:length(prev_state.X_ba);
        prev_state.P_ba = [prev_state.P_ba; [X_add_id(notL), ones(length(find(notL==1)),1)*i, P_add(notL,:)]]; %adding only those points whose 3d coord are new
        prev_state.P_ba = sortrows(prev_state.P_ba);
    end


    prev_state.prev_img = query_image;
    
    prev_state.D = [D_new;D_search]; %D_new contains those candidate keypts which were not outliers, D_search - those outlier which were not triangulated
    prev_state.E = [E_new;E_search];
    prev_state.To = [To_new;To_search];
    
    prev_state.A = A_new;
    prev_state.C = C_new;
    prev_state.F = F_new;
    prev_state.X = [worldPoints; X_add];
    prev_state.X_id = [X_id; X_add_id];
    prev_state.P = [points1; P_add];
    prev_state.n_landmark = [prev_state.n_landmark; length(prev_state.X(:,1))];
    
    L = ismembertol(C_new, F_new, 0.008,'ByRows',true);
    points_plot = C_new(~L,:);
    L = ismembertol(D_new, E_new, 0.008,'ByRows',true);
    points_plot = [points_plot; D_new(~L,:)];
    
    frame = plot_screencast(prev_state, points_plot);
    pause(0.1);    
    
    disp(["Current World points", size(prev_state.X,1)]);
end
figure