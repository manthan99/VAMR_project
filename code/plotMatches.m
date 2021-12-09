function plotMatches(matches, query_keypoints, database_keypoints)

[~, query_indices, match_indices] = find(matches);

x_from = query_keypoints(2, :);
x_to = database_keypoints(2, :);
y_from = query_keypoints(1, :);
y_to = database_keypoints(1, :);
plot([y_from; y_to], [x_from; x_to], 'g-', 'Linewidth', 3);

end

