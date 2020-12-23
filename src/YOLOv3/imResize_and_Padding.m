function [resized_img, tform] = imResize_and_Padding(img, rsz)
%% assume image processing toolbox is installed for using imwarp
%% imwarp関数を使うため、image processing toolboxが必要になります。

% int配列の場合singleにする
if isinteger(img)
    img = im2single(img);
end

% 目的サイズがスカラ入力の場合
if isscalar(rsz)
    rsz = [rsz rsz];
end

sz = size(img,[1:2]);

% 倍率の決定
[ratio, idx] = min(rsz ./ sz);

% ratio倍した後の画像のサイズ
scaled_sz = round(sz * ratio);

% paddingが必要な次元
pad_dim = (3-idx); % idx=1 then pad_dim=2, idx=2 then pad_dim=1;

% paddingの量（両側にpaddingする。半分だけの量を求める）
pad_s = round((rsz(pad_dim) - scaled_sz(pad_dim))/2);

% affine変換行列の定義
affineM = [ratio 0 0;0 ratio 0;0 0 1]; % スケール
affineM(3,idx) = pad_s; % シフト量
tform = affine2d(affineM);

% 幾何変換
resized_img = imwarp(img, tform, 'cubic', 'OutputView', imref2d(rsz), 'FillValues', 0.5);
end