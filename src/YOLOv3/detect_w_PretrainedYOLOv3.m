function [I, bboxesResized, scores, labels] = detect_w_PretrainedYOLOv3(img, yolov3net)
%% outputs
% I: input image with detected bounding boxes
% bboxesResized: bounding box data for original input image
% scores: scores for each detected bounding box
% labels: labels for each detected objects

%% load network
% load('yolov3-416.mat') ; % yolov3net

%% load class info. (80classes - COCO dataset)
classes = coco_classes();

%% resize image
inputImIdx = getImageInputLayerIndex(yolov3net);
inputImSize = yolov3net.Layers(inputImIdx).InputSize(1);
[I, tform] = imResize_and_Padding(img, inputImSize);

%% YOLOv3 Activation
% Get prediction at 3 different scales from YOLOv3
predictions = activations(yolov3net, I, 'concatenate_all');

%% post-processing
[bboxes, scores, labels] = yolov3DetectionGen(predictions, I, inputImSize);

%% visualization
label_str = {};
if ~isempty(bboxes)
    % bbox resize for original input image
    bboxesResized = bboxwarp(round(bboxes), invert(tform), imref2d(size(img)), 'OverlapThreshold', 0.1);
    bboxesResized = int32(round(bboxesResized));
    for ii=1:size(scores,1)
        class = char(classes{labels(ii,1),1});
        label_str{ii} = char(['Conf: ' num2str(scores(ii)*100,'%0.1f') '%, ' class]);
    end
    I = insertObjectAnnotation(img, 'rectangle', bboxesResized, label_str, ...
        'TextBoxOpacity', 0.9, 'FontSize', 12);
else
    I = img;
    bboxesResized = [];
end

end
%% Supporting Functions
function classes = coco_classes
    classes = cell(80,1);
    classes{1} = 'person';
    classes{2} = 'bicycle';
    classes{3} = 'car';
    classes{4} = 'motorbike';
    classes{5} = 'aeroplane';
    classes{6} = 'bus';
    classes{7} = 'train';
    classes{8} = 'truck';
    classes{9} = 'boat';
    classes{10} = 'traffic light';
    classes{11} = 'fire hydrant';
    classes{12} = 'stop sign';
    classes{13} = 'parking meter';
    classes{14} = 'bench';
    classes{15} = 'bird';
    classes{16} = 'cat';
    classes{17} = 'dog';
    classes{18} = 'horse';
    classes{19} = 'sheep';
    classes{20} = 'cow';
    classes{21} = 'elephant';
    classes{22} = 'bear';
    classes{23} = 'zebra';
    classes{24} = 'giraffe';
    classes{25} = 'backpack';
    classes{26} = 'umbrella';
    classes{27} = 'handbag';
    classes{28} = 'tie';
    classes{29} = 'suitcase';
    classes{30} = 'frisbee';
    classes{31} = 'skis';
    classes{32} = 'snowboard';
    classes{33} = 'sports ball';
    classes{34} = 'kite';
    classes{35} = 'baseball bat';
    classes{36} = 'baseball glove';
    classes{37} = 'skateboard';
    classes{38} = 'surfboard';
    classes{39} = 'tennis racket';
    classes{40} = 'bottle';
    classes{41} = 'wine glass';
    classes{42} = 'cup';
    classes{43} = 'fork';
    classes{44} = 'knife';
    classes{45} = 'spoon';
    classes{46} = 'bowl';
    classes{47} = 'banana';
    classes{48} = 'apple';
    classes{49} = 'sandwich';
    classes{50} = 'orange';
    classes{51} = 'broccoli';
    classes{52} = 'carrot';
    classes{53} = 'hot dog';
    classes{54} = 'pizza';
    classes{55} = 'donut';
    classes{56} = 'cake';
    classes{57} = 'chair';
    classes{58} = 'sofa';
    classes{59} = 'pottedplant';
    classes{60} = 'bed';
    classes{61} = 'diningtable';
    classes{62} = 'toilet';
    classes{63} = 'tvmonitor';
    classes{64} = 'laptop';
    classes{65} = 'mouse';
    classes{66} = 'remote';
    classes{67} = 'keyboard';
    classes{68} = 'classes phone';
    classes{69} = 'microwave';
    classes{70} = 'oven';
    classes{71} = 'toaster';
    classes{72} = 'sink';
    classes{73} = 'refrigerator';
    classes{74} = 'book';
    classes{75} = 'clock';
    classes{76} = 'vase';
    classes{77} = 'scissors';
    classes{78} = 'teddy bear';
    classes{79} = 'hair drier';
    classes{80} = 'toothbrush';
end

function idx = getImageInputLayerIndex(net)
for i = 1:length(net.Layers)
    layer = net.Layers(i);
    if isa(layer,'nnet.cnn.layer.ImageInputLayer')
        idx = i;
        break;
    end
end

end
function [bboxes, scores, labels] = yolov3DetectionGen(predictions, img, inputImSize)

%% 結果の分離
% Separate predictions
sz = size(predictions);
scale3 = predictions(1:sz(1), 1:sz(2), 511:end);
scale2 = predictions(1:sz(1)/2, 1:sz(2)/2, 256:510);
scale1 = predictions(1:sz(1)/4, 1:sz(2)/4, 1:255);

%% デコード用パラメータ設定
% Set parameters for post-processing
imSz = size(img);
% Image size
inputW = inputImSize;
inputH = inputImSize;
% クラス数(COCO dataset). 
numClasses = 80;
% X,Y,W,H,Score (5 elements).
numElements = 5;
% Ancers per Scale
numAnchors = 3;
% AnchorBoxes(3 different scales)
Anchors1 = [116,90;156,198;373,326];
Anchors2 = [30,61;62,45;59,119];
Anchors3 = [10,13;16,30;33,23];
% スコアに対する検出閾値
thresh = 0.5;
nms = 0.5;

%% ネットワーク出力配列を並び替え (ex. 13x13x255 > (13x13x3) x 85)
% Transform Predictions Array( Box Co-ordinates, Objectness Score and Class Scores )
fmap1 = transformPredictions(scale1, numAnchors, numClasses, numElements);
fmap2 = transformPredictions(scale2, numAnchors, numClasses, numElements);
fmap3 = transformPredictions(scale3, numAnchors, numClasses, numElements);

%% シグモイド関数を適用し、取り得るレンジに制約をかける
% Apply sigmoid to constraint its possible offset range
fmap1 = applySigmoid(fmap1);
fmap2 = applySigmoid(fmap2);
fmap3 = applySigmoid(fmap3);

%% Bounding Box算出
% Calculate Bounding Box
sz1 = size(scale1);
sz2 = size(scale2);
sz3 = size(scale3);
% 画像サイズ、アンカーの情報からBounding Box計算
fmap1 = calculateDetections(sz1,fmap1,Anchors1,inputW,inputH);
fmap2 = calculateDetections(sz2,fmap2,Anchors2,inputW,inputH);
fmap3 = calculateDetections(sz3,fmap3,Anchors3,inputW,inputH);

%% 異なるスケールにおける検出結果を統合
% Marge Detections
detections = [fmap1;fmap2;fmap3];

%% 閾値以上の結果を保持
% Filtering with the threshold
objectnessTmp = detections(:,5); %スコアの情報のみ抽出
detections = detections(objectnessTmp>thresh,:);
objectness = detections(:,5);
bboxes = [];

% 閾値以上のBoxがある場合
if ~isempty(detections)
    classProbs = detections(:,6:end);
    tmpSz = size(classProbs,2);
    tmpObj = repmat(objectness,1,tmpSz);
    classProbs = classProbs.*tmpObj;
    
    idx = classProbs > thresh;
    classProbs(~idx) = single(0);
    
    [idxa,idxb,probs] = find(classProbs);
    if size(classProbs,1)==1
        detections = [detections(idxa',1:4),probs',idxb'];
    else
        detections = [detections(idxa,1:4),probs,idxb];
    end
    if ~isempty(detections)
        % BBox用座標抽出
        % Extract tx, ty, tw and th for Bounding Box
        bboxes = [detections(:,1),detections(:,2),detections(:,3),detections(:,4)];
        
        % Bounding Boxのサイズを元画像に合わせる
        % Scale the size of BBox to align with imput image size
        bboxes = scaleBboxes(bboxes,imSz(1:2));
        
        bboxes = convertToXYWH(bboxes);
        scores = detections(:,5);
        labels = detections(:,6:end);
        
        % 境界部分のクリップ
        % Clip the bounding box when it is positioned outside the image
        bboxes = vision.internal.detector.clipBBox(bboxes, imSz(1:2));
        
        % 一応、ネガティブなBboxが存在する場合に削除しておく
        idx = all(bboxes>=1,2);
        bboxes = bboxes(idx,:);
        scores = scores(idx,:);
        labels = labels(idx,:);
        
        if ~isempty(bboxes)
            % 検出結果のマージ
            % Nonmaximal suppression to eliminate overlapping bounding boxes
            [bboxes, scores, labels] = selectStrongestBboxMulticlass(bboxes, scores, labels ,...
                'RatioType', 'Union', 'OverlapThreshold', nms);
        end
    end
end
if isempty(bboxes) 
    bboxes = [];
    scores = [];
    labels = [];
end
end
%% Supporting Functions
function tPred = transformPredictions(fmap,numAnchors,numClasses,numElements)
sz = size(fmap);
%13x13x255
tmpArray = permute(fmap,[2,1,3]);
%169x85x3
tmpArray = reshape(tmpArray,sz(1)*sz(2),numClasses+numElements,numAnchors);
%85x169x3
tmpArray = permute(tmpArray,[2,1,3]);
tmpSz = size(tmpArray);
%85x507
tmpArray = reshape(tmpArray,tmpSz(1),tmpSz(2)*tmpSz(3));
%507x85
tPred = permute(tmpArray,[2,1,3]);

end

function sPred = applySigmoid(tPred)

sigmoid = @(x) 1./(1+exp(-x));
xy = sigmoid(tPred(:,1:2));
wh = exp(tPred(:,3:4));
scores = sigmoid(tPred(:,5:end));

sPred = [xy, wh, scores];

end

function fmap = calculateDetections(sz,fmap,anchors,inputW,inputH)

base = [0:sz(2)-1]';
numAnchors = size(anchors,1);

colIndexes = repmat(repmat(base,sz(1),1),numAnchors,1);
rowIndexes = repmat(repelem(base,sz(1),1),numAnchors,1);
anchors = repelem(anchors,sz(1)*sz(2),1);

x = (colIndexes + fmap(:,1))./sz(2);
y = (rowIndexes + fmap(:,2))./sz(1);
w = fmap(:,3).*anchors(:,1)./inputW;
h = fmap(:,4).*anchors(:,2)./inputH;
r = fmap(:,5:end);

fmap = [x,y,w,h,r];
end

% 中心座標(x, y)をボックス左上の座標(X, Y)となるように変換
function dets = convertToXYWH(dets)
dets(:,1) = dets(:,1)- dets(:,3)/2 + 0.5;
dets(:,2) = dets(:,2)- dets(:,4)/2 + 0.5;
end

function bboxes = scaleBboxes(bboxes,imSz)
bboxes = [bboxes(:,1).*imSz(1,2),bboxes(:,2).*imSz(1,1),bboxes(:,3).*imSz(1,2),bboxes(:,4).*imSz(1,1)];
end

% Copyright 2020 The MathWorks, Inc.