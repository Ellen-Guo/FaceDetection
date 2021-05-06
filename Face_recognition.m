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
    imagegallery = montage(facedatabase(i).ImageLocation(1));
end

%% Splitting Database into Training and Testing sets
[training, testing] = partition(facedatabase,[0.8 0.2]);
image(training)
image(testing)

%% Extract and display Histogram of Oriented Gradient Features for Single Face
person = 5; 
[hogFeature, visualization] = extractHOGFeatures(read(training(person),1));
figure;
subplot (1,2,1); imshow(read(training(person),1)); title('Input Image');
subplot (1,2,2); plot(visualization);title('HOG Feature');

%% Extract HOG Features for Training set
trainingFeatures = zeros(size(training,2)*training(1).Count,4680);
featureCount = 1;
for i = 1:size(training,2)
    for j = 1:training(i).Count
        trainingFeatures(featureCount,:) = extractHOGFeatures(read(training(i),1));
        trainingLabel{featureCount} = training(i).Description;
        featureCount = featureCount+1;
    end 
    personIndex{i} = training(i).Description;
end

%% Create 40 class classifier using fitcecoc
faceClassifier = fitcecoc(trainingFeatures,trainingLabel);

%% Test Image from Test set
person = 1;
queryImage = read(testing(person),1);
queryFeatures = extractHOGFeatures(queryImage);
personLabel = predict(faceClassifier,queryFeatures);
booleanIndex = strcmp(personLabel, personIndex);
integerIndex = find(booleanIndex);
subplot(1,2,1); imshow(queryImage);title('Query Face');
subplot(1,2,2); imshow(read(training(integerIndex),1));title('Match Class');

%% 