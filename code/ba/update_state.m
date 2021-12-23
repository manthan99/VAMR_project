function [A_new, C_new, F_new, To_new, D_new, E_new, prev_state] = update_state(query_image, points1, R1, T1, harris_vars, prev_state)
% query_harris = harris(query_image, harris_vars.harris_patch_size, harris_vars.harris_kappa);
% C_new = selectKeypoints(...
%     query_harris, harris_vars.num_keypoints, harris_vars.nonmaximum_supression_radius);
% C_new = flipud(C_new)';
% C_new = detectMinEigenFeatures(query_image, 'MinQuality' , 0.005);
% C_new = C_new.Location;
C_new = detect_features(harris_vars, query_image);

D_new = C_new;

L = ismembertol(C_new, points1, 0.008,'ByRows',true);
L = ~L;
C_new = C_new(L,:); %keypoints that are candidates in query_image but not tracked currently

tracker = vision.PointTracker('MaxBidirectionalError',1);
initialize(tracker, prev_state.C, prev_state.prev_img);

[C_matched, inliers] = tracker(query_image); %tracking previous state candidates in current frame

C_matched = C_matched(inliers,:);
C_old = prev_state.C(inliers,:);
F_old = prev_state.F(inliers,:);
A_old = prev_state.A(inliers,:);

[L,Locb] = ismembertol(C_new, C_matched, 0.008,'ByRows',true);
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

tracker = vision.PointTracker('MaxBidirectionalError',1);
initialize(tracker, prev_state.D, prev_state.prev_img);

[D_matched, inliers] = tracker(query_image);

D_matched = D_matched(inliers,:);
D_old = prev_state.D(inliers,:);
E_old = prev_state.E(inliers,:);
To_old = prev_state.To(inliers,:);

[L,Locb] = ismembertol(D_new, D_matched, 0.008,'ByRows',true);
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