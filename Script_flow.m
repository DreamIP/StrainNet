clc 
clear all
close all

% If you do not use the star frames: 
    % 1) Set the good names of the csv files  *_disp_x.csv and *_disp_y.csv
    % 2) Change or comment the command caxis([-0.5 0.5]) 

disp_x = csvread('Star_disp_x.csv'); 
disp_y = csvread('Star_disp_y.csv');


figure
subplot(2,1,1)
imagesc(disp_x)
colormap('parula')
colorbar
caxis([-0.5 0.5])  
xlabel('x [pixel]')
ylabel('y [pixel]')
title('x-displacements')

subplot(2,1,2)
imagesc(disp_y)
colormap('parula')
colorbar
caxis([-0.5 0.5])
xlabel('x [pixel]')
ylabel('y [pixel]')
title('y-displacements')
