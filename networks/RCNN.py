import torch
import torch.nn as nn


class CNN_Block(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(CNN_Block, self).__init__()
        self.conv = nn.Conv3d(in_chn, out_chn, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.bn = nn.BatchNorm3d(out_chn)
    
    def forward(self, x):
        return self.bn(self.maxpool(self.relu(self.conv(x))))
        

class CNN_Extractor(nn.Module):
    def __init__(self):
        super(CNN_Extractor, self).__init__()
        self.conv_block1 = CNN_Block(in_chn=1, out_chn=32)
        self.conv_block2 = CNN_Block(in_chn=32, out_chn=64)
        self.conv_block3 = CNN_Block(in_chn=64, out_chn=128)
    
    def forward(self, x):
        return self.conv_block3(self.conv_block2(self.conv_block1(x)))


class RCNN(nn.Module):
    def __init__(self, input_size=13824, hidden_size=128, layer_num=2, window_size=25, stride=5):
        # rnn的input_size = 128 * 3 * 6 * 6 = 13824
        super(RCNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.window_size = window_size
        self.stride = stride

        self.cnn_extractor = CNN_Extractor()
        self.rnn = nn.GRU(input_size, self.hidden_size, self.layer_num, dropout=0.5, bidirectional=True)
        self.type_classifier = nn.Linear(self.hidden_size * 2, 4)
        self.stenosis_classifier = nn.Linear(self.hidden_size * 2, 3)

    def forward(self, x, steps, device):
        # steps为滑块个数。训练时为10，验证测试时为5。
        batch_size = x.size(0)
        rnn_input = torch.zeros(steps, batch_size, self.input_size).to(device)
        h0 = torch.zeros(2 * self.layer_num, batch_size, self.hidden_size)
        h0 = nn.init.orthogonal_(h0).to(device)
        for i in range(steps):
            input = x[:, :, i * self.stride: i * self.stride + self.window_size, :, :]
            rnn_input[i] = self.cnn_extractor(input).view(batch_size, -1)

        output, hn = self.rnn(rnn_input, h0)
        type_logits = self.type_classifier(output[-1])
        stenosis_logits = self.stenosis_classifier(output[-1])
        return type_logits, stenosis_logits
        

if __name__ == "__main__":
    rcnn = RCNN()
    device = torch.device('cpu')
    
    train_tensor = torch.randn(8, 1, 70, 50, 50)  # (N, C, D, H, W)
    type_pred_train, stenosis_pred_train = rcnn(train_tensor, 10, device)
    print(type_pred_train.shape, stenosis_pred_train.shape)

    val_tensor = torch.randn(8, 1, 45, 50, 50)  # (N, C, D, H, W)
    type_pred_val, stenosis_pred_val = rcnn(val_tensor, 5, device)
    print(type_pred_val.shape, stenosis_pred_val.shape)