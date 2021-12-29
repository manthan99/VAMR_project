%% Setup
clear all;
close all;

video_file = 'video_file_name_without_extension';

ds = 1; % 0: KITTI, 1: Malaga, 2: parking

ba_bool = true; %Bundle adjustment boolean variable
ba_n = 50; %Bundle adjustment window size
harris_vars = struct;
harris_vars.harris_patch_size = 9;
harris_vars.harris_kappa = 0.08;
harris_vars.nonmaximum_supression_radius = 9;
harris_vars.descriptor_radius = 9;
harris_vars.match_lambda = 5;
harris_vars.num_keypoints = 1000;

klt_vars = struct;
klt_vars.block = [31,31];
klt_vars.bidir_error = 1;

ds_vars = get_ds_vars(ds);


%% Bootstrap
% need to set bootstrap_frames
rng(1);
    
[inlierCurrPts, worldPoints, R1, T1, R, T, currImg, ds_vars, i] = bootstrap(ds_vars, harris_vars, klt_vars);

%% Continuous Operation Setup

%P X C F T setup for the first frame

%for BA function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X_ba = worldPoints; %X_ba will store all 3d points ever found
P_ba = [linspace(1,size(worldPoints,1),size(worldPoints,1))', ones(size(worldPoints,1),1)*(i-1), inlierCurrPts]; % columns - X_id, view_id, x, y

ViewId = uint32(i-1); 
Orientation = mat2cell(R, 3);
Location = mat2cell(T, 1);
pose_table_ba = table(ViewId, Orientation, Location);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

prev_img = currImg;
P_prev = inlierCurrPts;
X_prev = worldPoints;
X_id = (1:size(X_prev,1))';

corners0 = detect_features(harris_vars, prev_img);

D_prev = corners0;
E_prev = corners0;
%Remove Duplicates from corner0 corresponding to P_prev
[L,Locb] = ismembertol_Custom(corners0, P_prev, 0.008);
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
prev_state.frame = i-1;
prev_state.n_landmark = [size(prev_state.X,1)];

%% Continuous operation

range = i:ds_vars.last_frame;

error1_count = 0;
v = VideoWriter([video_file,'.avi']);
v.FrameRate=5;
open(v);
for i = range
    fprintf('\n\nProcessing frame %d\n=====================\n', i);
    
    query_image = load_image(ds_vars, i);
    prev_state.P_ba = prev_state.P_ba(find(prev_state.P_ba(:,2)>i-ba_n), :); %deleting old views 2d pts

    %find pose
    [R, T, worldPoints, X_id, outlier_id, points1, points_outliers] = findpose(query_image, ds_vars, prev_state, klt_vars);
    R_viz = R; %current camera pose
    T_viz = T;
    [R1,T1] = cameraPoseToExtrinsics(R,T);
    
    %BA vars update %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    prev_state.P_ba = [prev_state.P_ba; [X_id, ones(size(X_id,1),1)*i, points1]];
    prev_state.P_ba = sortrows(prev_state.P_ba);
    Orientation = mat2cell(R,3);
    Location = mat2cell(T,1);
    ViewId = uint32(i);
    prev_state.pose_table_ba = [prev_state.pose_table_ba; table(ViewId, Orientation, Location)];

    % Bundle adjustment %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ba_bool
        x_id = unique(prev_state.P_ba(:,1));% vector of all x ids
        hist_xid = histc(prev_state.P_ba(:,1),x_id);
        x_id = x_id(1<hist_xid); %those x_ids which have more than 1 2d points correspondences
        x_ba = prev_state.X_ba(x_id,:);
        point_tracks = [];
        pk=1;
        for j=x_id'
            pts = prev_state.P_ba(find(prev_state.P_ba(:,1)==j), 2:4);  %2d points corresponding to jth 3d point
            if pk==1
                point_tracks = [point_tracks; pointTrack(pts(:,1), pts(:,2:3))];
                point_tracks = repmat(point_tracks, size(x_id,1), 1);
            else
                point_tracks(pk) = [pointTrack(pts(:,1), pts(:,2:3))];
            end
            pk = pk+1;
        end
        [xyzRefinedPoints,refinedPoses, reproj_errors] = bundleAdjustment(x_ba, point_tracks, prev_state.pose_table_ba, ds_vars.intrinsics); %length of 3d points and pointtrack array should be same, right?
        prev_state.X_ba(x_id,:) = xyzRefinedPoints;
        prev_state.pose_table_ba = refinedPoses;
        worldPoints = prev_state.X_ba(X_id,:);
        R = cell2mat(prev_state.pose_table_ba.Orientation(end));
        T = cell2mat(prev_state.pose_table_ba.Location(end));
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [R1,T1] = cameraPoseToExtrinsics(R,T);
    
    [A_new, C_new, F_new, To_new, D_new, E_new, prev_state] = update_state(query_image, points1, R1, T1, harris_vars, prev_state, klt_vars);
    
    [X_orig, X_old_add, P_add, P_old_add, X_orig_id, X_old_add_id, D_search, E_search, To_search, A_new, C_new, F_new, D_new, E_new, To_new] = triangulate_new(A_new, C_new, F_new, To_new, D_new, E_new, prev_state, worldPoints, points_outliers, outlier_id, ds_vars, R1, T1);
    
    %prev_state update %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    prev_state.X_ba = [prev_state.X_ba; X_orig];
    prev_state.P_ba = [prev_state.P_ba; [X_orig_id, ones(size(X_orig_id,1),1)*i, P_add]];
    prev_state.P_ba = sortrows(prev_state.P_ba);

    prev_state.prev_img = query_image;
    
    prev_state.D = [D_new;D_search]; %D_new contains those candidate keypts which were not outliers, D_search - those outlier which were not triangulated
    prev_state.E = [E_new;E_search];
    prev_state.To = [To_new;To_search];
    
    prev_state.A = A_new;
    prev_state.C = C_new;
    prev_state.F = F_new;
    prev_state.X = [worldPoints; X_orig; X_old_add];
    prev_state.X_id = [X_id; X_orig_id; X_old_add_id];
    prev_state.P = [points1; P_add; P_old_add];
    prev_state.n_landmark = [prev_state.n_landmark; size(prev_state.X(:,1),1)];
    prev_state.frame = i;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    L = ismembertol_Custom(C_new, F_new, 0.008);
    points_plot = C_new(~L,:);
    L = ismembertol_Custom(D_new, E_new, 0.008);
    points_plot = [points_plot; D_new(~L,:)];
    
    frame = plot_screencast(prev_state, points_plot);
    writeVideo(v, frame2im(getframe(frame)));
    pause(0.1);    
    
    disp(["Current World points", size(prev_state.X,1)]);
end
close(v);
figure