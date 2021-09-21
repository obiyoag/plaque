import torch
from torch import nn, einsum
from einops import rearrange, repeat
from networks.cnn_extractors import CNN_Extractor_3D, CNN_Extractor_2D


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

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

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class TR_Net_3D(nn.Module):
    def __init__(self, window_size, stride, steps, input_size=13824, dim=1024, depth=12, pool='cls'):
        # rnn的input_size = 128 * 3 * 6 * 6 = 13824
        super(TR_Net_3D, self).__init__()
        self.input_size = input_size
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.stride = stride
        self.steps = steps

        self.cnn_extractor = CNN_Extractor_3D()
        self.to_embedding = nn.Linear(input_size, dim)

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pos_embedding = nn.Parameter(torch.randn(1, self.steps + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(0.1)
        self.transformer = Transformer(dim, depth, heads=16, dim_head=64, mlp_dim=2048, dropout=0.1)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.type_classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4))
        self.stenosis_classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3))

    def forward(self, x, device):
        # steps为滑块个数。训练时为10，验证测试时为5。
        batch_size = x.size(0)
        transformer_input = torch.zeros(batch_size, self.steps, self.input_size).to(device)
        for i in range(self.steps):
            input = x[:, :, i * self.stride: i * self.stride + self.window_size, :, :]
            transformer_input[:, i, :] = self.cnn_extractor(input).view(batch_size, -1)
        transformer_input = self.to_embedding(transformer_input)
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=batch_size)
        transformer_input = torch.cat((cls_tokens, transformer_input), dim=1)
        transformer_input += self.pos_embedding[:, :(self.steps + 1)]
        transformer_input = self.dropout(transformer_input)

        output = self.transformer(transformer_input)

        output = output.mean(dim=1) if self.pool == 'mean' else output[:, 0]

        output = self.to_latent(output)
        type_logits = self.type_classifier(output)
        stenosis_logits = self.stenosis_classifier(output)
        return type_logits, stenosis_logits


class TR_Net_2D(nn.Module):
    def __init__(self, window_size, stride, steps, input_size=4608, dim=1024, depth=12, pool='cls'):
        # rnn_2d的input_size = 128 * 6 * 6 = 4608
        super(TR_Net_2D, self).__init__()
        self.window_size = window_size
        self.stride = stride
        self.input_size = input_size
        self.dim = dim
        self.depth = depth
        self.steps = steps

        self.cnn_extractor = CNN_Extractor_2D(in_chn=window_size)
        self.to_embedding = nn.Linear(input_size, dim)

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pos_embedding = nn.Parameter(torch.randn(1, self.steps + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(0.1)
        self.transformer = Transformer(dim, depth, heads=16, dim_head=64, mlp_dim=2048, dropout=0.1)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.type_classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4))
        self.stenosis_classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3))

    def forward(self, x, device):
        # steps为滑块个数。训练时为10，验证测试时为5。
        batch_size = x.size(0)
        transformer_input = torch.zeros(batch_size, self.steps, self.input_size).to(device)
        for i in range(self.steps):
            input = x[:, :, i * self.stride: i * self.stride + self.window_size, :, :].squeeze(1)
            transformer_input[:, i, :] = self.cnn_extractor(input).view(batch_size, -1)
        transformer_input = self.to_embedding(transformer_input)
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=batch_size)
        transformer_input = torch.cat((cls_tokens, transformer_input), dim=1)
        transformer_input += self.pos_embedding[:, :(self.steps + 1)]
        transformer_input = self.dropout(transformer_input)

        output = self.transformer(transformer_input)

        output = output.mean(dim=1) if self.pool == 'mean' else output[:, 0]

        output = self.to_latent(output)
        type_logits = self.type_classifier(output)
        stenosis_logits = self.stenosis_classifier(output)
        return type_logits, stenosis_logits


if __name__ == "__main__":
    rcnn = TR_Net_2D(5, 2)
    device = torch.device('cpu')
    
    train_tensor = torch.randn(8, 1, 45, 50, 50)  # (N, C, D, H, W)
    type_pred_train, stenosis_pred_train = rcnn(train_tensor, 5, device)
    print(type_pred_train.shape, stenosis_pred_train.shape)
