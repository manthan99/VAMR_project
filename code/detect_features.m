function corners0 = detect_features(harris_vars, prev_img, ds_vars)
% Detects harris features
% Inputs: harris_vars: parameters for harris feature detection, prev_img: image
% Outputs: corners0: harris features detected in image

if ds_vars.ds==0
    corners0 = detectHarrisFeatures(prev_img);
    corners0 = corners0.Location;
else
    query_harris = harris(prev_img, harris_vars.harris_patch_size, harris_vars.harris_kappa);
    corners0 = selectKeypoints(...
        query_harris, harris_vars.num_keypoints, harris_vars.nonmaximum_supression_radius);
    corners0 = flipud(corners0)';
end

end