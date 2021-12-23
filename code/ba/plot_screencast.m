function [frame] = plot_screencast(prev_state, C)
ax1 = subplot(2,4,[1,2], 'color', [1,1,1]);
imshow(prev_state.prev_img, 'Parent', ax1);
hold on
plot(prev_state.P(:,1), prev_state.P(:,2), 'g+', 'Parent', ax1);
plot(C(:,1), C(:,2), 'ro', 'Parent', ax1);
title(['Current image, frame # ',num2str(prev_state.frame)], 'Parent', ax1);
hold off

ax2 = subplot(2,4,5);
if length(prev_state.n_landmark)<20
    plot(linspace(-length(prev_state.n_landmark)+1,0,length(prev_state.n_landmark)), prev_state.n_landmark, 'Parent', ax2);
else
    plot(linspace(-19,0,20), prev_state.n_landmark(end-19:end), 'Parent', ax2);
end
title('# tracked landmarks over last 20 frames', 'Parent', ax2);

ax3 = subplot(2,4,6);
% plot(squeeze(prev_state.Trajectory(1,4,:)), squeeze(prev_state.Trajectory(3,4,:)), 'b-', 'Parent', ax3);
% hold on
loc = cell2mat(prev_state.pose_table_ba.Location);
plot(loc(:,1), loc(:,3), 'r-', 'Parent', ax3, 'LineWidth',2, 'LineStyle',':');
title('Full Trajectory', 'Parent', ax3);
% axis equal
axis padded
% hold off

% ax4 = subplot(2,4,[3,4,7,8]);
% plot(prev_state.X(:,1), prev_state.X(:,3), 'k.', 'Parent', ax4);
% hold on
% if length(prev_state.Trajectory(1,4,:))<20
%     plot(squeeze(prev_state.Trajectory(1,4,:)), squeeze(prev_state.Trajectory(3,4,:)), 'b', 'Parent', ax4);
% else
%     plot(squeeze(prev_state.Trajectory(1,4,end-19:end)), squeeze(prev_state.Trajectory(3,4,end-19:end)), 'b', 'Parent', ax4);
% end
% title('Trajectory of last 20 frames', 'Parent', ax4);
% hold off

ax4 = subplot(2,4,[3,4,7,8]);
plot(prev_state.X(:,1), prev_state.X(:,3), 'k.', 'Parent', ax4);
hold on
if length(loc(:,1))<20
    plot(loc(:,1), loc(:,3), 'b', 'Parent', ax4, 'LineWidth',2, 'LineStyle','-');
else
    plot(loc(end-19:end,1), loc(end-19:end,3), 'b', 'Parent', ax4, 'LineWidth',2, 'LineStyle','-');
end
title('Trajectory of last 20 frames', 'Parent', ax4);
hold off

set(gcf,'color','w');
frame = gcf;
end