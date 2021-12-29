function ds_vars = get_ds_vars(ds)
% Returns dataset parameters

ds_vars = struct;
ds_vars.ds = ds; % 0: KITTI, 1: Malaga, 2: parking

if ds_vars.ds == 0
    % need to set kitti_path to folder containing "05" and "poses"
    ds_vars.path = '../../data/kitti05/kitti';
    ds_vars.ground_truth = load([ds_vars.path, '/poses/05.txt']);
    ds_vars.ground_truth = ds_vars.ground_truth(:, [end-8 end]);
    ds_vars.last_frame = 4540;
    ds_vars.K = [7.188560000000e+02 0 6.071928000000e+02
        0 7.188560000000e+02 1.852157000000e+02
        0 0 1];

elseif ds_vars.ds == 1
    % Path containing the many files of Malaga 7.
    ds_vars.path = '../../data/malaga-urban-dataset-extract-07';
    ds_vars.images = dir([ds_vars.path,'/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    ds_vars.left_images = ds_vars.images(3:2:end);
    ds_vars.last_frame = length(ds_vars.left_images);
    ds_vars.K = [621.18428 0 404.0076
        0 621.18428 309.05989
        0 0 1];
elseif ds_vars.ds == 2
    % Path containing images, depths and all...
    ds_vars.path = '../../data/parking';
    ds_vars.last_frame = 598;
    ds_vars.K = load([ds_vars.path, '/K.txt']);
    ds_vars.ground_truth = load([ds_vars.path, '/poses.txt']);
    ds_vars.ground_truth = ds_vars.ground_truth(:, [end-8 end]);
else
    assert(false);
end
end