import torch
import torch.nn as nn
from networks.cnn_extractors import CNN_Extractor_2D, CNN_Extractor_3D


class RCNN_3D(nn.Module):
    def __init__(self, window_size, stride, steps, input_size=13824, hidden_size=128, layer_num=2):
        # rnn的input_size = 128 * 3 * 6 * 6 = 13824
        super(RCNN_3D, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.window_size = window_size
        self.stride = stride
        self.steps = steps

        self.cnn_extractor = CNN_Extractor_3D()
        self.rnn = nn.GRU(input_size, self.hidden_size, self.layer_num, dropout=0.5, bidirectional=True)
        self.type_classifier = nn.Linear(self.hidden_size * 2, 4)
        self.stenosis_classifier = nn.Linear(self.hidden_size * 2, 3)

    def forward(self, x, device):
        # steps为滑块个数。训练时为10，验证测试时为5。
        batch_size = x.size(0)
        rnn_input = torch.zeros(self.steps, batch_size, self.input_size).to(device)
        h0 = torch.zeros(2 * self.layer_num, batch_size, self.hidden_size)
        h0 = nn.init.orthogonal_(h0).to(device)
        for i in range(self.steps):
            input = x[:, :, i * self.stride: i * self.stride + self.window_size, :, :]
            rnn_input[i] = self.cnn_extractor(input).view(batch_size, -1)

        output, hn = self.rnn(rnn_input, h0)
        type_logits = self.type_classifier(output[-1])
        stenosis_logits = self.stenosis_classifier(output[-1])
        return type_logits, stenosis_logits
        

class RCNN_2D(nn.Module):
    def __init__(self, window_size, stride, steps, input_size=4608, hidden_size=128, layer_num=2):
        # rnn_2d的input_size = 128 * 6 * 6 = 4608
        super(RCNN_2D, self).__init__()
        self.window_size = window_size
        self.stride = stride
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.steps = steps

        self.cnn_extractor = CNN_Extractor_2D(in_chn=window_size)
        self.rnn = nn.GRU(input_size, self.hidden_size, self.layer_num, dropout=0.5, bidirectional=True)
        self.type_classifier = nn.Linear(self.hidden_size * 2, 4)
        self.stenosis_classifier = nn.Linear(self.hidden_size * 2, 3)

    def forward(self, x, device):
        # steps为滑块个数。训练时为10，验证测试时为5。
        batch_size = x.size(0)
        rnn_input = torch.zeros(self.steps, batch_size, self.input_size).to(device)
        h0 = torch.zeros(2 * self.layer_num, batch_size, self.hidden_size)
        h0 = nn.init.orthogonal_(h0).to(device)
        for i in range(self.steps):
            input = x[:, :, i * self.stride: i * self.stride + self.window_size, :, :].squeeze(1)
            rnn_input[i] = self.cnn_extractor(input).view(batch_size, -1)

        output, hn = self.rnn(rnn_input, h0)
        type_logits = self.type_classifier(output[-1])
        stenosis_logits = self.stenosis_classifier(output[-1])
        return type_logits, stenosis_logits


if __name__ == "__main__":
    rcnn = RCNN_2D(1, 1)
    device = torch.device('cpu')
    
    train_tensor = torch.randn(8, 1, 17, 50, 50)  # (N, C, D, H, W)
    type_logits, stenosis_logits = rcnn(train_tensor, 17, device)
    print(type_logits.shape, stenosis_logits.shape)
