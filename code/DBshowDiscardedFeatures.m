% Shows which features where identified as the same feature and therefore
% discardd

function DBshowDiscardedFeatures(image, newCandidatePoints, existingPoints, logicalTrueNewPoints)
    
    DBNewPoints = size(newCandidatePoints,1)
    DBTrackedPoints = size(existingPoints,1)
    DBEstiamtedRemainingPoints= DBNewPoints - DBTrackedPoints
    DBActualRemainingPointsOld = sum(logicalTrueNewPoints)

    figure;
    imshow(image); hold on;
    scatter(newCandidatePoints(:,1),newCandidatePoints(:,2));
    scatter(existingPoints(:,1),existingPoints(:,2),'green','+');
    scatter(newCandidatePoints(~logicalTrueNewPoints,1),newCandidatePoints(~logicalTrueNewPoints,2),'red','x');
    legend(sprintf('New Candidates: %.1f ',DBNewPoints), ...
           sprintf('Tracked Points: %.1f ',DBTrackedPoints), ...
           sprintf('Discarded Points: %.1f ', sum(~logicalTrueNewPoints)));
end