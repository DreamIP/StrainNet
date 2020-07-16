clc
clear all
close all

% create Train_data and Test_data directories
mkdir Train_Data
mkdir Test_data

addpath('Reference_speckle_frames'); % change the path if the Reference frames are not in the default directory
addpath('Train_Data');
addpath('Test_data');

%Train dataset 
n=100;  

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
    
    % Region of 8x8 
    xp0=[1:1/8:SubsetSize/8+1-1/8]+2;
    yp0=[1:1/8:SubsetSize/8+1-1/8]+2;
    xxp0=1:(SubsetSize/8)+3;
    yyp0=1:(SubsetSize/8)+3;

    [Xp_subset0,Yp_subset0] = meshgrid(xp0,yp0);

       for l = 1:n
       
       % A random displamcent for each region 8x8 
        f =  randi([-100 100],SubsetSize/8+3,SubsetSize/8+3)/100;  
        g =  randi([-100 100],SubsetSize/8+3,SubsetSize/8+3)/100;      
        
        % linear interpolation between the randoom displacments 
        x0 = Xp_subset0 ;
        y0 = Yp_subset0 ;
        disp_x = interp2(xxp0,yyp0,f,x0,y0,'linear');
        disp_y = interp2(xxp0,yyp0,g,x0,y0,'linear');
        
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
        name_ref = sprintf('Train_Data/Ref%03d_%03d.csv',img,l);   % '/Test_data/...'
        name_def = sprintf('Train_Data/Def%03d_%03d.csv',img,l);   % '/Test_data/...'
        name_dispx = sprintf('Train_Data/Dispx%03d_%03d.csv',img,l); % '/Test_data/...'
        name_dispy = sprintf('Train_Data/Dispy%03d_%03d.csv',img,l); % '/Test_data/...'
        
        %  Write the data 
        dlmwrite(name_ref, Image_Ref, 'delimiter', ',', 'precision', '%.0f');
        dlmwrite(name_def, Image_BD, 'delimiter', ',', 'precision', '%.3f');
        dlmwrite(name_dispx, disp_x, 'delimiter', ',', 'precision', '%.2f');
        dlmwrite(name_dispy, disp_y, 'delimiter', ',', 'precision', '%.2f');

     end 
  end 

%Test dataset 
 n=1; 

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
    
    % Region of 8x8 
    xp0=[1:1/8:SubsetSize/8+1-1/8]+2;
    yp0=[1:1/8:SubsetSize/8+1-1/8]+2;
    xxp0=1:(SubsetSize/8)+3;
    yyp0=1:(SubsetSize/8)+3;

    [Xp_subset0,Yp_subset0] = meshgrid(xp0,yp0);

       for l = 1:n
       
       % A random displamcent for each region 8x8 
        f =  randi([-100 100],SubsetSize/8+3,SubsetSize/8+3)/100;  
        g =  randi([-100 100],SubsetSize/8+3,SubsetSize/8+3)/100;      
        
        % linear interpolation between the randoom displacments 
        x0 = Xp_subset0 ;
        y0 = Yp_subset0 ;
        disp_x = interp2(xxp0,yyp0,f,x0,y0,'linear');
        disp_y = interp2(xxp0,yyp0,g,x0,y0,'linear');
        
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
        name_ref = sprintf('Test_data/Ref%03d_%03d.csv',img,l); 
        name_def = sprintf('Test_data/Def%03d_%03d.csv',img,l);  
        name_dispx = sprintf('Test_data/Dispx%03d_%03d.csv',img,l); 
        name_dispy = sprintf('Test_data/Dispy%03d_%03d.csv',img,l); 
                
        %  Write the data 
        dlmwrite(name_ref, Image_Ref, 'delimiter', ',', 'precision', '%.0f');
        dlmwrite(name_def, Image_BD, 'delimiter', ',', 'precision', '%.3f');
        dlmwrite(name_dispx, disp_x, 'delimiter', ',', 'precision', '%.2f');
        dlmwrite(name_dispy, disp_y, 'delimiter', ',', 'precision', '%.2f');

     end 
  end 
