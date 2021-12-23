function [X_orig, X_old_add, P_add, P_old_add, X_orig_id, X_old_add_id, D_search, E_search, To_search, A_new, C_new, F_new, D_new, E_new, To_new] = triangulate_new(A_new, C_new, F_new, To_new, D_new, E_new, prev_state, worldPoints, points_outliers, outlier_id, ds_vars, R1, T1)
X_orig = []; %new world points added
P_add = []; %new keypoints added corresponding to X
X_orig_id = [];
X_old_add = [];
P_old_add = [];
X_old_add_id = [];
rem_idx = [];
D_search = [];
E_search = [];
To_search = [];

%Now check for new triangulations
if size(worldPoints,1) < 400

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
        minParallax = 0.5; % In degrees
        A = reshape(A_mat(i,:),3,4);
        R = A(1:3,1:3);
        T = A(1:3,4)';
        cam_mat_0 = cameraMatrix(ds_vars.intrinsics, R, T);
        cam_mat_1 = cameraMatrix(ds_vars.intrinsics, R1, T1);

        %             if(T == T1 | end_ind - start_ind < 10)
        if(T == T1)
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

        X_orig = [X_orig; XYZpts(prlx_ind,:)];
        P_add = [P_add; Cpts(prlx_ind,:)];
        rem_idx = [rem_idx, rem_ind(prlx_ind)];

    end

    [C_new,PS] = removerows(C_new,'ind',rem_idx);
    [F_new,PS] = removerows(F_new,'ind',rem_idx);
    [A_new,PS] = removerows(A_new,'ind',rem_idx);

    disp(["New points triangulated", size(X_orig,1)]);
    X_orig_id = [(size(prev_state.X_ba,1)+1:size(prev_state.X_ba,1)+size(X_orig,1))'];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Find the dropped outliers and only check for their triangulations
    rem_idx = [];
    [L,locb] = ismembertol_Custom(D_new, points_outliers, 0.008);
    D_search_id = outlier_id(locb(L));
    D_search = D_new(L,:);
    E_search = E_new(L,:);
    To_search = To_new(L,:);
    D_new = D_new(~L,:);
    E_new = E_new(~L,:);
    To_new = To_new(~L,:);
    % Get blocks of  with same A
    [To_mat,ia,ic] = unique(To_search,'rows','stable');

    for i=1:size(ia,1)
        start_ind = ia(i);

        if i < size(ia,1)
            end_ind = ia(i+1);
        else
            end_ind = size(To_search,1);
        end

        % Triangulate two views to obtain 3-D map points
        minParallax = 0.5; % In degrees
        A = reshape(To_mat(i,:),3,4);
        R = A(1:3,1:3);
        T = A(1:3,4)';
        cam_mat_0 = cameraMatrix(ds_vars.intrinsics, R, T);
        cam_mat_1 = cameraMatrix(ds_vars.intrinsics, R1, T1);

        %             if(T == T1 | end_ind - start_ind < 10)
        if(T == T1)
            %Skip the iteration since they are the current points
            continue
        end

        [XYZpts,reprojectionErrors,isInFront] = triangulate(E_search(start_ind:end_ind,:), D_search(start_ind:end_ind,:),cam_mat_0,cam_mat_1);

        % Filter points by view direction and reprojection error
        minReprojError = 2;
        inlierIdx  = isInFront & reprojectionErrors < minReprojError;
        XYZpts  = XYZpts(inlierIdx ,:);
        Dpts = D_search(start_ind:end_ind,:);
        Dpts_id = D_search_id(start_ind:end_ind);
        Dpts = Dpts(inlierIdx ,:);
        Dpts_id = Dpts_id(inlierIdx);

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

        X_old_add = [X_old_add; XYZpts(prlx_ind,:)];
        P_old_add = [P_old_add; Dpts(prlx_ind,:)];
        X_old_add_id = [X_old_add_id; Dpts_id(prlx_ind)];
        rem_idx = [rem_idx, rem_ind(prlx_ind)];

    end

    [D_search,PS] = removerows(D_search,'ind',rem_idx);
    [E_search,PS] = removerows(E_search,'ind',rem_idx);
    [To_search,PS] = removerows(To_search,'ind',rem_idx);

    disp(["New points added from outliers", size(X_old_add,1)]);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
end