close all
clear 
clc
%% Loading images from the database
facedatabase = imageSet('FaceDatabase','recursive');

%% Display Montage of First Face
figure;
montage(facedatabase(1).ImageLocation);
title('Image of single face');

%% Display Query Image and Database side to side
personToQuery = 30;
galleryImage = read(facedatabase(personToQuery),1);
figure;
for i = 1:size(facedatabase,2)
    imagegallery{i} = facedatabase(i).ImageLocation(1);
%     imds = imageDatastore(imagegallery);
%     imageLabeler(imds);
end
subplot(1,2,1); imshow(galleryImage); title("Query Image");
subplot(1,2,2);montage(string(imagegallery));title("All Image");


%% Splitting Database into Training and Testing sets
% [training, testing] = partition(facedatabase,[0.8 0.2]);
PD = 0.80;
N = size(facedatabase,1);
idx = 1;
training = facedatabase(idx(1:round(N*PD)),:);
testing = facedatabase(idx(round(N*PD):end),:);

%% Extract and display Histogram of Oriented Gradient Features for Single Face
person = 5; 
%[hogFeature, visualization] = extractHOGFeatures(read(training(person),1));
[hogFeature, visualization] = HOG(read(training(person),1));
H = HOGpicture(visualization,9);
figure;
subplot (1,2,1); imshow(read(training(person),1)); title('Input Image');
subplot (1,2,2); imshow(H); title('histogram image');
%subplot (1,2,2); plot(visualization);title('HOG Feature');

%% Extract HOG Features for Training set
trainingFeatures = zeros(size(training,2)*training(1).Count,2520);
featureCount = 1;
for i = 1:size(training,2)
    for j = 1:training(i).Count
        %trainingFeatures(featureCount,:) = extractHOGFeatures(read(training(i),1));
        %[trainingFeatures,trainingvisualization] = HOG(read(training(i),1));
        trainingFeatures(featureCount,:) = HOG(read(training(i),1));
        trainingLabel{featureCount} = training(i).Description;
        featureCount = featureCount+1;
    end 
    personIndex{i} = training(i).Description;
end

%% Create 40 class classifier using fitcecoc
faceClassifier = fitcecoc(trainingFeatures,trainingLabel);

%% Testing for a single image
person = 40;
[queryFeatures,queryVisualization] = HOG(read(testing(person),1));
personLabel = predict(faceClassifier,queryFeatures);
booleanIndex = strcmp(personLabel, personIndex);
integerIndex = find(booleanIndex);
figure;
sgtitle('Singular testing');
subplot(1,2,1); imshow(read(testing(person),1));title('Query Face');
subplot(1,2,2); imshow(read(training(integerIndex),1));title('Match Class');


%% Test Image from Test set
counter = 0;
position = 0;
figure;
% sgtitle('Testing set of 5 images');
for person = 1:5
    p = 10;
    i = randsample(p,1);
    p1 = 10;
    randomp = randsample(p1,1);
    [queryFeatures,queryVisualization] = HOG(read(testing(randomp),i));
    personLabel = predict(faceClassifier,queryFeatures);
    booleanIndex = strcmp(personLabel, personIndex);
    integerIndex = find(booleanIndex);
    counter = counter+1;
    count = num2str(counter);
    position = position+1;
    subplot(5,2,position); imshow(imresize(read(testing(randomp),i),3));title(append('Query Image ', count));
    position = position+1;
    subplot(5,2,position); imshow(read(training(integerIndex),1));title(append('Match Class ',training(integerIndex).Description));
    %disp(training(integerIndex));
end


%% HOG filter implementation
% img = imread('girl.tif');
% [F, M] = HOG(img);
% H = HOGpicture(M, 9);
% [featureVector,hogVisualization] = extractHOGFeatures(img);
% figure
% subplot(1,2,1)
% imshow(H);
% subplot(1,2,2)
% plot(hogVisualization);

function [feature, matrix] = HOG(image)
[mu, theta] = grad(image); %gradient mag. and ang. of image
theta = Angle(theta); %angle readjustment; negative to positive angle
[m,n] = size(image);
% find # of blocks
v_blocks = floor(floor(m/16) + (floor(m/16)/2));
h_blocks = floor(floor(n/16) + (floor(n/16)/2));
matrix = zeros(v_blocks, h_blocks, 9);

feature = []; % feature vector
delta = 20;
for i = 1:v_blocks
    for k = 1:h_blocks
        Bx = i*8;
        By = k*8;
        % 16 x 16 blocks; overlapping
        block_mu = mu((Bx-7):(Bx+8), (By-7):(By+8));
        block_ang = theta((Bx-7):(Bx+8), (By-7):(By+8));
        block = []; % block feature
        for u = 1:2
            for v = 1:2
                hist = zeros(1,9); %histogram bins
                Cx = u*8;
                Cy = v*8;
                % 8x8 cells
                cell_mu = block_mu((Cx-7):Cx, (Cy-7):Cy);
                cell_ang = block_ang((Cx-7):Cx, (Cy-7):Cy);
                for x = 1:8
                    for y = 1:8
                        ang = cell_ang(x,y);
                        mag = cell_mu(x,y);
                        j = floor((ang/20) - (1/2));
                        if j < 0
                            j = 0;
                        end
                        j_center = bin_c(j);
                        j1_center = bin_c(j+1);
                        hist(1, j+1) = hist(1, j+1) + mag*((j1_center - ang)/delta);
                        hist(1, j+2) = hist(1, j+2) + mag*((ang - j_center)/delta);
                    end
                end
                block = [block hist];
            end
        end
        block = block / sqrt(norm(block)^2 + .01); %normalize the block vector L1-Norm
        B1 = reshape(block, 2, 2, 9);
        matrix(i:i+1, k:k+1, :) = B1;
        feature = [feature block];
    end
end
%normalize the feature vector L2-Norm
feature = feature / sqrt(norm(feature)^2 + .001);
feature(feature > 0.2) = 0.2;
feature = feature / sqrt(norm(feature)^2 + .001);
end
% Gradient angle and mag calculations
function [mag, angle] = grad(A)
    Gx = [-1 0 1];
    Gy = Gx';
    
    % centered hor. and ver. gradient
    gx = double(imfilter(A, Gx));
    gy = double(imfilter(A, Gy));
    
    % magnitude and angle
    angle = atan2(gy, gx) .* 180/pi;
    mag = (gx.^2 + gy.^2).^(1/2);
end
% Angle Readjustment
function g = Angle(A)
    [m,n] = size(A);
    for x = 1:m
        for y = 1:n
            ang = A(x,y);
            if ang < 0
                A(x,y) = 180 + ang;
            end
        end
    end
    g = A;
end
% Bin center
function z = bin_c(j)
z = 20*(j + (1/2));
end
% Visualize the HOGFeature *** Code for Github ***
function im = HOGpicture(w, bs)
    % Make picture of positive HOG weights.
    %   im = HOGpicture(w, bs)

    % construct a "glyph" for each orientation
    bim1 = zeros(bs, bs);
    bim1(:,round(bs/2):round(bs/2)+1) = 1;
    bim = zeros([size(bim1) 9]);
    bim(:,:,1) = bim1;
    for i = 2:9
      bim(:,:,i) = imrotate(bim1, -(i-1)*20, 'crop');
    end

    % make pictures of positive weights bs adding up weighted glyphs
    s = size(w);    
    w(w < 0) = 0;    
    im = zeros(bs*s(1), bs*s(2));
    for i = 1:s(1)
     iis = (i-1)*bs+1:i*bs;
     for j = 1:s(2)
        jjs = (j-1)*bs+1:j*bs;          
       for k = 1:9
         im(iis,jjs) = im(iis,jjs) + bim(:,:,k) * w(i,j,k);
      end
     end
    end
end
