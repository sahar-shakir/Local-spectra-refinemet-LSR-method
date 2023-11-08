%This is the main implementation of the method proposed in:
% 'Unsupervised fabric defect detection with local spectra refinement (LSR'
% by Sahar Shakir and Cihan Topal

If you use this code please cite as:
Shakir, S., Topal, C. Unsupervised fabric defect detection with local spectra refinement (LSR). Neural Comput & Applic (2023). https://doi.org/10.1007/s00521-023-09080-0.
%
% This code uses 32x32 window size
% The input image size 256x256
% The ground truth info is extracted from text file
% The only parameter need to be modified is the threshold value
% Each fabric image is classified as defected or defect free
% Localiation of defects is also available

clear ;
clc;

% Read images from directory
imagefiles = dir('*.png');
nfiles = length(imagefiles);

% thresold value unique for each fabric type
Threshold = 70000;

% % Loop for each image
for s=1:nfiles
    % Read image from disk and parse names
    currentfilename = imagefiles(s).name;
    currentimage = imread(currentfilename);
    [pathstr,name,ext] = fileparts(currentfilename);
    % read ground truth from text
    filename = [name,'.txt'];
    fileID = fopen(filename,'r');
    ground = fscanf(fileID,'%d');
    fclose(fileID);
    gtimage = ground(5);
    if gtimage == 00
        gt(s)=00;
    else
        gt(s)=01;
    end
    
    % Convert image to gray
    InputGray=rgb2gray(currentimage);
    IMAGE_selected=double(InputGray);
    [M,N]= size(IMAGE_selected);
    
    patch_No = 64;
    VAR_Image=cell(patch_No,4);
    step_x=16;
    step_y=16;
    Msub=32;
    Nsub=32;
    step_x_ref = 32;
    step_y_ref = 32;
    half_patches=32;
    VAR_patch_det = cell(patch_No,4);
    defect_x_y = zeros(patch_No,4);
    NEW_AVERAGE_fft_mag = zeros(Msub,Nsub);
    count=0;
    count_det = 0;
    fourier_image =zeros(M,N);
    count_defect=0;
    count_patches =0;
    FourierAverageWN(Msub,Nsub)=0;
    
    %constructing refernce (average image)
    %Sliding window by step_x through column of image
    for x_ref = 1 : step_x_ref : M
        % Sliding window by step_y through row of image
        for y_ref = 1 : step_y_ref : N
            differenceMAT_Avg=zeros(Msub,Nsub);
            SW_ref=imcrop(IMAGE_selected,[x_ref y_ref Msub-1 Nsub-1]);
            [SW_mag_ref ]= fourier(SW_ref);
            FourierAverageWN = FourierAverageWN + SW_mag_ref ;
            count_patches = count_patches + 1 ;
            
        end
    end
    %Take average of all elements in matrix
    FourierAverage = FourierAverageWN / (count_patches) ;
    
    %         figure(1);
    %         range_of_visualization = [0 5000];
    %         fft_shift_first_average= fftshift(FourierAverage);
    %         imshow(fft_shift_first_average,range_of_visualization);
    
    % compare patches and average matrix to optimize refernce image
    New_patch_fft(Msub,Nsub)=0;
    SW = zeros(Msub,Nsub);
    
    for xx = 1 : step_x_ref : M
        for yy = 1 : step_y_ref : N
            SW=imcrop(IMAGE_selected,[xx yy  Msub-1 Nsub-1]);
            %fourier transform of patch
            [New_patch_fft]= fourier(SW); 
            %Subtract average matrix from FFT matrix of selected patch for average optimization
            for m = 1: Msub
                for n = 1 : Nsub
                    differenceMAT_Avg(m,n) = New_patch_fft(m,n) - FourierAverage(m,n);
                end
            end
        
            count = count + 1 ;
            Var_Patch = var(differenceMAT_Avg(:));
        
            %Store a single value as variance of our patch in a vector
            VAR_Image{count,1} = Var_Patch;
            VAR_Image{count,2} = count ;
            VAR_Image{count,3} = xx ;
            VAR_Image{count,4} = yy ;
        end
    end
    VAR_Image_sorted = sortrows(VAR_Image);
    
    %eliminating  half of the patches and reconstructing reference images from other half
    for count_sorted_var = 1 : half_patches
        x_sort = VAR_Image_sorted{count_sorted_var,3};
        y_sort = VAR_Image_sorted{count_sorted_var,4};
        New_patch=imcrop(IMAGE_selected,[x_sort y_sort Msub-1 Nsub-1]);
        [New_patch_fft]= fourier(New_patch);
        NEW_AVERAGE_fft_mag = NEW_AVERAGE_fft_mag + New_patch_fft ;
    end
    AVERAGE_MAT = NEW_AVERAGE_fft_mag / half_patches;
    
    %     shifted= fftshift(AVERAGE_MAT);
    %     range_of_visualization = [0 5000];
    %     figure(2)
    %     imshow(shifted,[0 5000]);
    
    % % % % % % %% segmentation of patchs and detection
    for x_patch = 1 : step_x : M - step_x
        for y_patch = 1 : step_y : N - step_y  
            Detected_patch = zeros(Msub,Nsub);
            Detected_patch_fft = zeros(Msub,Nsub);
            differenceMAT_Det = zeros(Msub,Nsub);
            Detected_patch=imcrop(IMAGE_selected,[x_patch y_patch  Msub-1 Nsub-1]);
            [Detected_patch_fft]= fourier(Detected_patch);
            %implementing differnce matrix betwwen refined average and selected patch
            for m_det = 1: Msub
                for n_det = 1 : Nsub
                    differenceMAT_Det(m_det,n_det) = Detected_patch_fft(m_det,n_det) - AVERAGE_MAT(m_det,n_det);
                end
            end            
            
            count_det = count_det + 1 ;
            Var_Patch_detected = var(differenceMAT_Det(:));
            VAR_patch_det{count_det,1} = Var_Patch_detected;
            VAR_patch_det{count_det,2} = count_det ;
            VAR_patch_det{count_det,3} = x_patch ;
            VAR_patch_det{count_det,4} = y_patch ;
            %Detect if each patch is defective or clean
            if Var_Patch_detected > Threshold
                defect_x_y(count_defect+1,1)= x_patch ;
                defect_x_y(count_defect+1,2)= y_patch;
                defect_x_y(count_defect+1,3)= Msub ;
                defect_x_y(count_defect+1,4)= Nsub ;
                count_defect = count_defect + 1;
            end
        end
    end
    
    %draw the image with defects detected
    if defect_x_y(1,1) ~= 0
        defect(s)=1;
                RGB = insertShape(currentimage,'Rectangle',(defect_x_y),'Color','yellow','Opacity',1);
                figure('name','LSC Method')
                imshow(RGB);
    else
        defect(s)=0;
                figure('name','LSC Method');
                imshow(currentimage);
    end
    clearvars -except imagefiles currentimage gt defect nfiles count_accuracy count_threshold Threshold accuracyThreshold ;
end



