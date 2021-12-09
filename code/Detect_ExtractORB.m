function [features, valid_pts] = Detect_ExtractORB(Img, ...
    scaleFactor, numLevels, numPoints)

%Convert to grayscale if not already converted
Img_mono  = im2gray(Img);

%Detect the ORB features
keypoints = detectORBFeatures(Img_mono, 'ScaleFactor', scaleFactor, 'NumLevels', numLevels);

% Select numPoints uniformly distributed features across the image 
keypoints = selectUniform(keypoints, numPoints, size(Img_mono));

% Extract ORB features
[features, valid_pts] = extractFeatures(Img_mono, keypoints);


end