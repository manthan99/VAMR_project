function result = Add_ORB(points0, points1)

if(isempty(points0) & ~isempty(points1))
    result = points1;
    return;
end

if(isempty(points1) & ~isempty(points0))
    result = points0;
    return;
end    

if(isempty(points1) & isempty(points0))
    result = [];
    return;
end

locations = [points0.Location;points1.Location];
scales = [points0.Scale; points1.Scale];
angles = [points0.Orientation; points1.Orientation];
result = ORBPoints(locations,'Scale',scales,'Orientation',angles);
 
end