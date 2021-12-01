import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


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


class ViT(nn.Module):
    def __init__(self, window_size, image_size=(50, 50), dim=1024, depth=12, heads=16, mlp_dim=2048, pool='cls', dim_head=64, dropout=0.1, emb_dropout=0.1):
        super(ViT, self).__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.frame_to_embedding = nn.Sequential(
            Rearrange('b c d h w -> b d (c h w)'),
            nn.Linear(image_size[0]*image_size[1], dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, window_size + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.type_classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4))
        self.stenosis_classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3))
    
    def forward(self, img, device):
        # x = rearrange(img, 'b c d h w -> b d (c h w)')
        x = self.frame_to_embedding(img)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=img.size(0))
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(img.size(2) + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        type_logits = self.type_classifier(x)
        stenosis_logits = self.stenosis_classifier(x)
        return type_logits, stenosis_logits


if __name__ == "__main__":
    device = torch.device('cuda')
    vit = ViT(window_size=45).cuda()
    train_tensor = torch.randn(8, 1, 45, 50, 50).cuda()  # (N, C, D, H, W)
    type_pred_train, stenosis_pred_train = vit(train_tensor, device)
    print(type_pred_train.shape, stenosis_pred_train.shape)
