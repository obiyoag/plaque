import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadDotProductAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = (dim / heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        _, _, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Encoder1DBlock(nn.Module):
    def __init__(self, input_shape, heads, mlp_dim, dtype=torch.float32, dropout_rate=0.1, attention_dropout_rate=0.1,
                 deterministic=True):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.dtype = dtype
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.deterministic = deterministic
        self.input_shape = input_shape
        self.layer_norm_input = nn.LayerNorm(input_shape)
        self.layer_norm_out = nn.LayerNorm(input_shape)

        # self.layer_norm_input = nn.GroupNorm(1)
        # self.layer_norm_out = nn.GroupNorm(1)

        self.attention = MultiHeadDotProductAttention(input_shape, heads=heads)
        self.mlp = FeedForward(input_shape, mlp_dim, dropout_rate)
        self.drop_out_attention = nn.Dropout(attention_dropout_rate)

    def forward(self, inputs):
        x = self.layer_norm_input(inputs)
        x = self.attention(x)
        x = self.drop_out_attention(x)
        x = x + inputs
        y = self.layer_norm_out(x)
        y = self.mlp(y)
        return x + y

class transformer_structure(nn.Module):
    def __init__(self, input_shape, num_layers, heads, mlp_dim, inputs_positions=None, dropout_rate=0.1, train=False):
        super().__init__()
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.inputs_positions = inputs_positions
        self.dropout_rate = dropout_rate
        self.train_flag = train
        self.encoder_norm = nn.LayerNorm(input_shape)
        # self.encoder_norm = nn.GroupNorm(1)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([Encoder1DBlock(input_shape, heads, mlp_dim)]))

    def forward(self, img):
        x = img
        for layer in self.layers:
            x = layer[0](x)
        return self.encoder_norm(x)

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Conv3d(nn.Module):
    def __init__(self, in_channels, num_levels=4, f_maps=16):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.in_channels = in_channels
        for i in range(num_levels):
            self.layers.append(conv3x3x3(self.in_channels, f_maps * (2 ** i), stride=1))
            self.layers.append(nn.BatchNorm3d(f_maps * (2 ** i)))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.MaxPool3d(kernel_size=(2, 2, 2), padding=1))
            self.in_channels = f_maps * (2 ** i)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = rearrange(x, 'b c h w l  -> b (c h w l)')
        return x

class softmax_classify(nn.Module):
    def __init__(self, hidden_size, n_classes, CLS_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(hidden_size, 864))
        self.layers.append(nn.Linear(864, 432))
        self.layers.append(nn.Linear(432, 216))
        self.layers.append(nn.Linear(216, 54))
        self.layers.append(nn.Linear(54, n_classes))

        self.soft_max = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(CLS_dropout)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        x = self.soft_max(x)
        return x

class transformer_network(nn.Module):
    def __init__(self, window_size, stride, steps,
                 in_channels=1, conv_levels=4, conv_maps=16, hidden_size=9600, encoder_num=8,
                 heads=12, mlp_dim=3072, dropout=0.1, emb_dropout=0.1, CLS_dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.steps = steps
        self.hidden_size = hidden_size
        self.order_embedding = nn.Parameter(torch.randn(1, self.steps, hidden_size))
        self.conv_layer = Conv3d(in_channels, num_levels=conv_levels, f_maps=conv_maps)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = transformer_structure(hidden_size, encoder_num, heads, mlp_dim, dropout_rate=dropout)
        self.type_classifier = softmax_classify(hidden_size, 4, CLS_dropout)
        self.stenosis_classifier = softmax_classify(hidden_size, 3, CLS_dropout)

    def forward(self, img, device):
        batch_size = img.size(0)
        transformer_input = torch.zeros(batch_size, self.steps, self.hidden_size).to(device)
        for i in range(self.steps):
            input = img[:, :, i * self.stride: i * self.stride + self.window_size, :, :]
            transformer_input[:, i, :] = self.conv_layer(input).view(batch_size, -1)
        transformer_input += self.order_embedding
        transformer_input = self.dropout(transformer_input)
        output = self.transformer(transformer_input)[:, 0]
        type_logits = self.type_classifier(output)
        stenosis_logits = self.stenosis_classifier(output)

        return type_logits, stenosis_logits


if __name__ == '__main__':
    TR_Net = transformer_network(25, 5, 9, in_channels=1, conv_levels=4, conv_maps=16,
                                 hidden_size=9600, encoder_num=8, heads=12,
                                 mlp_dim=3072, dropout=0.1, emb_dropout=0.1, CLS_dropout=0.1)
    img = torch.randn(8, 1, 65, 50, 50)
    type_logtis, stenosis_logits = TR_Net(img)
    print(type_logtis.shape, stenosis_logits.shape)
