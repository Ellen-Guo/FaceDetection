img = imread('girl.tif');
[F, M] = HOG(img);
H = HOGpicture(M, 9);
[featureVector,hogVisualization] = extractHOGFeatures(img);
figure
imshow(img);
title('Original');
figure
subplot(1,3,2)
imshow(H);
title('Self Implementation');
figure
plot(hogVisualization);
title('MATLAB Built-in');

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
%% Gradient angle and mag calculations
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
%% Angle Readjustment
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
%% Bin center
function z = bin_c(j)
z = 20*(j + (1/2));
end
%% Visualize the HOGFeature *** Code for Github by rbgirshick (Ross Girshick)***
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