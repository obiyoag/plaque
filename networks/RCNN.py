import torch
import torch.nn as nn
from networks.cnn_extractors import CNN_Extractor_2D, CNN_Extractor_3D


class RCNN(nn.Module):
    def __init__(self, input_size=13824, hidden_size=128, layer_num=2, window_size=25, stride=5):
        # rnn的input_size = 128 * 3 * 6 * 6 = 13824
        super(RCNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.window_size = window_size
        self.stride = stride

        self.cnn_extractor = CNN_Extractor_3D()
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
        

class RCNN_2D(nn.Module):
    def __init__(self, input_size=4608, hidden_size=128, layer_num=2, window_size=25, stride=5):
        # rnn_2d的input_size = 128 * 6 * 6 = 4608
        super(RCNN_2D, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.window_size = window_size
        self.stride = stride

        self.cnn_extractor = CNN_Extractor_2D()
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
            input = x[:, :, i * self.stride: i * self.stride + self.window_size, :, :].squeeze(1)
            rnn_input[i] = self.cnn_extractor(input).view(batch_size, -1)

        output, hn = self.rnn(rnn_input, h0)
        type_logits = self.type_classifier(output[-1])
        stenosis_logits = self.stenosis_classifier(output[-1])
        return type_logits, stenosis_logits


if __name__ == "__main__":
    rcnn = RCNN_2D()
    device = torch.device('cpu')
    
    train_tensor = torch.randn(8, 1, 70, 50, 50)  # (N, C, D, H, W)
    type_pred_train, stenosis_pred_train = rcnn(train_tensor, 10, device)
    print(type_pred_train.shape, stenosis_pred_train.shape)

    val_tensor = torch.randn(8, 1, 45, 50, 50)  # (N, C, D, H, W)
    type_pred_val, stenosis_pred_val = rcnn(val_tensor, 5, device)
    print(type_pred_val.shape, stenosis_pred_val.shape)

    train_tensor = torch.randn(8, 1, 70, 50, 50)  # (N, C, D, H, W)
    input_tensor = train_tensor[:, :, 0: 25, :, :].squeeze(1)
    print(train_tensor.shape)
    cnn = CNN_Extractor_2D()
    output = cnn(input_tensor)
    print(output.shape)