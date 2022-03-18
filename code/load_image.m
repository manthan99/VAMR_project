function image = load_image(ds_vars, i)
% Loads ith image from dataset specified by ds_vars

    if ds_vars.ds == 0
        image = imread([ds_vars.path '/05/image_0/' sprintf('%06d.png',i)]);
    elseif ds_vars.ds == 1
        image = rgb2gray(imread([ds_vars.path ...
            '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            ds_vars.left_images(i).name]));
    elseif ds_vars.ds == 2
        image = im2uint8(rgb2gray(imread([ds_vars.path ...
            sprintf('/images/img_%05d.png',i)])));
    else
        assert(false);
    end
end