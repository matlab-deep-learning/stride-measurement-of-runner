classdef ZeroPadding4YOLOv3Layer < nnet.layer.Layer

    properties
        Amounts
        Top
        Bottom
        Left
        Right        
    end
    
    methods
        function this = ZeroPadding4YOLOv3Layer(name, Amounts)
            this.Amounts= Amounts;
            this.Name = name;
            this.Description = ['Pad images with zeros for YOLOv3']
            this.Type = ['Zero padding 2d'];
        end
        
        function Z = predict( this, X )
            % X is size [H W C N]. Z is size [H+Top+Bottom, W+Left+Right, C, N].
            [H,W,C,N] = size(X);
            this.Top = 0;
            this.Left = 0;
            this.Bottom = H * (this.Amounts - 1);
            this.Right = W * (this.Amounts - 1);
            Z = zeros(H + this.Top + this.Bottom, W + this.Left + this.Right, C, N, 'like', X);
            Z(this.Top+(1:H), this.Left+(1:W), :, :) = X;
        end
        
        function dLdX = backward( this, X, Z, dLdZ, memory )
            % dLdX and X are size [H W C N]. dLdZ is size [H+Top+Bottom, W+Left+Right, C, N]. 
            [H,W,C,N] = size(X);
            this.Top = 0;
            this.Left = 0;            
            dLdX = dLdZ(this.Top+(1:H), this.Left+(1:W), :, :);
        end
    end
end
