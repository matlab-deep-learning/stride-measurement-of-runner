function net = finalizeImportedYOLOv3Lgraph(lgraph, inputImSize)
%% this function finalizes imported YOLOv3 layergraph by tb_darknet2ml_new.mlx

% error case
if inputImSize ~= 320 & inputImSize ~= 416 & inputImSize ~= 608
    error('inputImSize must be either 320, 416 or 608!!')
end

FeatureMapSize1 = inputImSize / 32;
FeatureMapSize2 = inputImSize / 16;
lastSize = inputImSize / 8;

%% replace inputImageLayer
lgraph = removeLayers(lgraph, 'input');
lgraph = addLayers(lgraph, imageInputLayer([inputImSize,inputImSize,3], 'Name', 'input', 'Normalization', 'none'));
lgraph = connectLayers(lgraph, 'input', 'conv2d_1');
%% integrate outputs
lgraph = addLayers(lgraph, ZeroPadding4YOLOv3Layer('zero_padding2d_l1', 4));
lgraph = addLayers(lgraph, ZeroPadding4YOLOv3Layer('zero_padding2d_l2', 2));
lgraph = connectLayers(lgraph, 'conv2d_59', 'zero_padding2d_l1');
lgraph = connectLayers(lgraph, 'conv2d_67', 'zero_padding2d_l2');
tempLayers = [
    depthConcatenationLayer(3,'Name','concatenate_all')
    regressionLayer('Name','output')];
lgraph = addLayers(lgraph,tempLayers);
lgraph = connectLayers(lgraph,'zero_padding2d_l1','concatenate_all/in1');
lgraph = connectLayers(lgraph,'zero_padding2d_l2','concatenate_all/in2');
lgraph = connectLayers(lgraph,'conv2d_75','concatenate_all/in3');

net = assembleNetwork(lgraph);
end
