function [frame] = plot_screencast(img, P, C, landmark_n, landmarks, Trajectory)
ax1 = subplot(2,4,[1,2], 'color', [1,1,1]);
imshow(img, 'Parent', ax1);
hold on
plot(P(:,1), P(:,2), 'g+', 'Parent', ax1);
plot(C(:,1), C(:,2), 'ro', 'Parent', ax1);
title('Current image', 'Parent', ax1);
hold off

ax2 = subplot(2,4,5);
if length(landmark_n)<20
    plot(linspace(-length(landmark_n)+1,0,length(landmark_n)), landmark_n, 'Parent', ax2);
else
    plot(linspace(-19,0,20), landmark_n(end-19:end), 'Parent', ax2);
end
title('# tracked landmarks over last 20 frames', 'Parent', ax2);

ax3 = subplot(2,4,6);
plot(squeeze(Trajectory(1,4,:)), squeeze(Trajectory(3,4,:)), 'LineWidth',2, 'Parent', ax3);
title('Full Trajectory', 'Parent', ax3);
axis equal
xlim padded
ylim padded

ax4 = subplot(2,4,[3,4,7,8]);
plot(landmarks(:,1), landmarks(:,3), 'k.', 'Parent', ax4);
hold on
if length(Trajectory(1,4,:))<20
    plot(squeeze(Trajectory(1,4,:)), squeeze(Trajectory(3,4,:)), 'b', 'LineWidth',2, 'Parent', ax4);
else
    plot(squeeze(Trajectory(1,4,end-19:end)), squeeze(Trajectory(3,4,end-19:end)), 'b', 'LineWidth',2, 'Parent', ax4);
end
title('Trajectory of last 20 frames', 'Parent', ax4);
axis equal
hold off

set(gcf,'color','w');
frame = gcf;
end