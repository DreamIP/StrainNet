clc
clear all
close all
 
% create Train_data and Test_data directories
mkdir Train_Data
mkdir Test_data

addpath('Reference_speckle_frames'); % change the path if the Reference frames are not in the default directory
addpath('Train_Data');
addpath('Test_data');


 
n=60;  %Train dataset


SubsetSize = 256;

parfor img = 1:363

    name_image = sprintf('Ref%01d.tif',img);
    Image_Ref = double(imread(name_image));
 
    Image_Ref_interpol = zeros(SubsetSize+3,SubsetSize+3);  % add padding of 0 for interpolation window 4x4 and translation [-1,1]
    Image_Ref_interpol(2:SubsetSize+1,2:SubsetSize+1)= Image_Ref;

    xp=[1:SubsetSize] + 1;
    yp=[1:SubsetSize] + 1;
    xxp=1:SubsetSize + 3;
    yyp=1:SubsetSize + 3;
    [Xp_subset,Yp_subset] = meshgrid(xp,yp);

    % Define the regions size 
    for l = 1:n
    if l <11 % l==1
        s = 128;
    elseif  l<21 % l==2
        s = 64;
    elseif  l<31 % l==3
        s = 32;
    elseif  l<41 % l==4
        s = 16;
    elseif  l<51 % l==5
        s = 8;
    else
        s = 4;
    end

    xp0=[1:1/s:SubsetSize/s+1-1/s]+2;
    yp0=[1:1/s:SubsetSize/s+1-1/s]+2;
    xxp0=1:(SubsetSize/s)+3;
    yyp0=1:(SubsetSize/s)+3;
    [Xp_subset0,Yp_subset0] = meshgrid(xp0,yp0);
    
    % A random displamcent for each region 
    f =  randi([-100 100],SubsetSize/s+3,SubsetSize/s+3)/115;
    g =  randi([-100 100],SubsetSize/s+3,SubsetSize/s+3)/115;
    
    % Bicubic interpolation between the randoom displacments 
    x0 = Xp_subset0 ;
    y0 = Yp_subset0 ;
    disp_x = interp2(xxp0,yyp0,f,x0,y0,'cubic');
    disp_y = interp2(xxp0,yyp0,g,x0,y0,'cubic');
        
     % Setting the boundries dispalcements to 0
    disp_x(1:2,:) = 0;
    disp_y(1:2,:) = 0;        
    disp_x(:,1:2) = 0;
    disp_y(:,1:2) = 0;        
    disp_x(SubsetSize-1:SubsetSize,:) = 0;
    disp_y(SubsetSize-1:SubsetSize,:) = 0;        
    disp_x(:,SubsetSize-1:SubsetSize) = 0;
    disp_y(:,SubsetSize-1:SubsetSize) = 0;   
        
     % Generate the deformed image based on bi-cubic interpolation
     x = Xp_subset + disp_x;
     y = Yp_subset + disp_y;
     Image_BD = interp2(xxp,yyp,Image_Ref_interpol,x,y,'cubic');

     % Data labels
     name_ref = sprintf('Train_Data/Ref%03d_%02d.csv',img,l);
     name_def = sprintf('Train_Data/Def%03d_%02d.csv',img,l); 
     name_dispx = sprintf('Train_Data/Dispx%03d_%02d.csv',img,l);
     name_dispy = sprintf('Train_Data/Dispy%03d_%02d.csv',img,l);

     %  Write the data
     dlmwrite(name_ref, Image_Ref, 'delimiter', ',', 'precision', '%.0f');
     dlmwrite(name_def, Image_BD, 'delimiter', ',', 'precision', '%.3f');
     dlmwrite(name_dispx, disp_x, 'delimiter', ',', 'precision', '%.3f');
     dlmwrite(name_dispy, disp_y, 'delimiter', ',', 'precision', '%.3f');

     end 
end 

%Test dataset
 n=6;
parfor img = 1:363

    name_image = sprintf('Ref%01d.tif',img);
    Image_Ref = double(imread(name_image));
 
    Image_Ref_interpol = zeros(SubsetSize+3,SubsetSize+3);  % add padding of 0 for interpolation window 4x4 and translation [-1,1]
    Image_Ref_interpol(2:SubsetSize+1,2:SubsetSize+1)= Image_Ref;

    xp=[1:SubsetSize] + 1;
    yp=[1:SubsetSize] + 1;
    xxp=1:SubsetSize + 3;
    yyp=1:SubsetSize + 3;
    [Xp_subset,Yp_subset] = meshgrid(xp,yp);

    % Define the regions size 
    for l = 1:n
    if l <11 % l==1
        s = 128;
    elseif  l<21 % l==2
        s = 64;
    elseif  l<31 % l==3
        s = 32;
    elseif  l<41 % l==4
        s = 16;
    elseif  l<51 % l==5
        s = 8;
    else
        s = 4;
    end

    xp0=[1:1/s:SubsetSize/s+1-1/s]+2;
    yp0=[1:1/s:SubsetSize/s+1-1/s]+2;
    xxp0=1:(SubsetSize/s)+3;
    yyp0=1:(SubsetSize/s)+3;
    [Xp_subset0,Yp_subset0] = meshgrid(xp0,yp0);
    
    % A random displamcent for each region 
    f =  randi([-100 100],SubsetSize/s+3,SubsetSize/s+3)/115;
    g =  randi([-100 100],SubsetSize/s+3,SubsetSize/s+3)/115;
    
    % Bicubic interpolation between the randoom displacments 
    x0 = Xp_subset0 ;
    y0 = Yp_subset0 ;
    disp_x = interp2(xxp0,yyp0,f,x0,y0,'cubic');
    disp_y = interp2(xxp0,yyp0,g,x0,y0,'cubic');
        
     % Setting the boundries dispalcements to 0
    disp_x(1:2,:) = 0;
    disp_y(1:2,:) = 0;        
    disp_x(:,1:2) = 0;
    disp_y(:,1:2) = 0;        
    disp_x(SubsetSize-1:SubsetSize,:) = 0;
    disp_y(SubsetSize-1:SubsetSize,:) = 0;        
    disp_x(:,SubsetSize-1:SubsetSize) = 0;
    disp_y(:,SubsetSize-1:SubsetSize) = 0;   
        
     % Generate the deformed image based on bi-cubic interpolation
     x = Xp_subset + disp_x;
     y = Yp_subset + disp_y;
     Image_BD = interp2(xxp,yyp,Image_Ref_interpol,x,y,'cubic');

     % Data labels
     name_ref = sprintf('Test_Data/Ref%03d_%02d.csv',img,l); 
     name_def = sprintf('Test_Data/Def%03d_%02d.csv',img,l); 
     name_dispx = sprintf('Test_Data/Dispx%03d_%02d.csv',img,l); 
     name_dispy = sprintf('Test_Data/Dispy%03d_%02d.csv',img,l); 

     %  Write the data
     dlmwrite(name_ref, Image_Ref, 'delimiter', ',', 'precision', '%.0f');
     dlmwrite(name_def, Image_BD, 'delimiter', ',', 'precision', '%.3f');
     dlmwrite(name_dispx, disp_x, 'delimiter', ',', 'precision', '%.3f');
     dlmwrite(name_dispy, disp_y, 'delimiter', ',', 'precision', '%.3f');

     end 
end
