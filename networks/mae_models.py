import torch
import torch.nn as nn
from networks.ViT import Transformer
from einops.layers.torch import Rearrange
from utils import get_sinusoid_encoding_table
from timm.models.layers import trunc_normal_


class PretrainVisionTransformerEncoder(nn.Module):
    def __init__(self, window_size, pretrain, image_size=(50, 50), dim=1024, depth=12, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1):
        super(PretrainVisionTransformerEncoder, self).__init__()
        self.dim = dim
        self.pretrain = pretrain
        self.frame_to_embedding = nn.Sequential(
            Rearrange('b c d h w -> b d (c h w)'),
            nn.Linear(image_size[0]*image_size[1], dim),
        )
        self.pos_embed = get_sinusoid_encoding_table(window_size, dim)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.norm = nn.LayerNorm(dim)
        
        if pretrain:
            self.type_classifier = nn.Identity()
            self.stenosis_classifer = nn.Identity()
        else:
            self.fc_norm = nn.LayerNorm(dim)
            self.type_classifier = nn.Linear(self.dim, 4)
            self.stenosis_classifer = nn.Linear(self.dim, 3)
            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, mask):
        x = self.frame_to_embedding(x)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        x_vis = x[~mask].reshape(x.size(0), -1, x.size(-1))
        x_vis = self.transformer(x_vis)
        x_vis = self.norm(x_vis)

        if self.pretrain:
            return x_vis
        else:
            x_vis = self.fc_norm(x.mean(1))
            return self.type_classifier(x_vis), self.stenosis_classifer(x_vis)


class PretrainVisionTransformerDecoder(nn.Module):
    def __init__(self, image_size=(50, 50), dim=1024, depth=12, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1):
        super(PretrainVisionTransformerDecoder, self).__init__()
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, image_size[0]*image_size[1])

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, return_token_num):
        x = self.transformer(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:]))
        else:
            x = self.head(self.norm(x))
        
        return x

class PretrainViT(nn.Module):
    def __init__(self, window_size, image_size=(50, 50), encoder_dim=1024, encoder_depth=12, encoder_heads=16,
                 decoder_dim=512, decoder_depth=8, decoder_heads=8):
        super(PretrainViT, self).__init__()
        self.encoder = PretrainVisionTransformerEncoder(window_size, True, image_size, encoder_dim, encoder_depth, encoder_heads)
        self.decoder = PretrainVisionTransformerDecoder(image_size, decoder_dim, decoder_depth, decoder_heads)
        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed = get_sinusoid_encoding_table(window_size, decoder_dim)

        trunc_normal_(self.mask_token, mean=0., std=0.02, a=-0.02, b=0.02)
    
    def forward(self, x, mask):
        x_vis = self.encoder(x, mask)
        x_vis = self.encoder_to_decoder(x_vis)

        B, N, C = x_vis.shape

        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accordingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)
        x = self.decoder(x_full, pos_emd_mask.shape[1])
        x = torch.sigmoid(x)

        return x
