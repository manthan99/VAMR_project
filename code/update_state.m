function [A_new, C_new, F_new, To_new, D_new, E_new, prev_state] = update_state(query_image, points1, R1, T1, harris_vars, prev_state, klt_vars, ds_vars)
% This functions finds new candidates for traingulation and structures them into variables of prev_state
% Inputs: query_image: current frame, points1: image points currently tracked
% R1,T1: transformation from world coordinates to camera coordinates
% harris_vars: parameters for harris feature detection, prev_state: struct for continuous operation states, klt_vars: parameters for klt

C_new = detect_features(harris_vars, query_image, ds_vars);

D_new = C_new;

L = ismembertol_Custom(C_new, points1, 0.008);
L = ~L;
C_new = C_new(L,:); %keypoints that are candidates in query_image but not tracked currently

tracker = vision.PointTracker('MaxBidirectionalError',klt_vars.bidir_error, 'BlockSize', klt_vars.block);
initialize(tracker, prev_state.C, prev_state.prev_img);

[C_matched, inliers] = tracker(query_image); %tracking previous state candidates in current frame

C_matched = C_matched(inliers,:);
C_old = prev_state.C(inliers,:);
F_old = prev_state.F(inliers,:);
A_old = prev_state.A(inliers,:);

[L,Locb] = ismembertol_Custom(C_new, C_matched, 0.008);
F_new = C_new;
A_new = [R1,T1'];
A_new = reshape(A_new,1,[]);
A_new = repmat(A_new,size(C_new,1),1);

F_new(L,:) = F_old(Locb(L),:); %setting 2d coord. of points that were there in C previously to be previous state coord
A_new(L,:) = A_old(Locb(L),:);

%Need to remove the duplicates of F (why will there be duplicates)
[F_new,ia,ic] = unique(F_new,'rows','stable');
A_new = A_new(ia,:);
C_new = C_new(ia,:);

%Sort according to get same Transformations together
[A_new,index] = sortrows(A_new);
C_new = C_new(index,:);
F_new = F_new(index,:);

%%%%%%%%%%%%%%%%%%%%%%%%% Now perform updates for D, E and To

tracker = vision.PointTracker('MaxBidirectionalError',klt_vars.bidir_error, 'BlockSize', klt_vars.block);
initialize(tracker, prev_state.D, prev_state.prev_img);

[D_matched, inliers] = tracker(query_image);

D_matched = D_matched(inliers,:);
D_old = prev_state.D(inliers,:);
E_old = prev_state.E(inliers,:);
To_old = prev_state.To(inliers,:);

[L,Locb] = ismembertol_Custom(D_new, D_matched, 0.008);
E_new = D_new;
To_new = [R1,T1'];
To_new = reshape(To_new,1,[]);
To_new = repmat(To_new,size(D_new,1),1);

E_new(L,:) = E_old(Locb(L),:);
To_new(L,:) = To_old(Locb(L),:);

%Need to remove the duplicates of E
[E_new,ia,ic] = unique(E_new,'rows','stable');
To_new = To_new(ia,:);
D_new = D_new(ia,:);

%Sort according to get same Transformations together
[To_new,index] = sortrows(To_new);
D_new = D_new(index,:);
E_new = E_new(index,:);
end