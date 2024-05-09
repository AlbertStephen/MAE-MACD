import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block


def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        patches = take_indexes(patches, forward_indexes)

        return patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle()
        self.ratio = mask_ratio
        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, patches):
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        return features

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=256,
                 patch_size=32,
                 emb_dim=512,
                 num_layer=8,
                 num_head=64,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature
        patches = self.head(features)
        mask = torch.zeros_like(patches)

        mask[T:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask


class Teacher_MAE(torch.nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 emb_dim=768,
                 encoder_layer=12,
                 encoder_head=12,
                 decoder_layer=8,
                 decoder_head=64,
                 latten_dim = 2048,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        # # [batch_size, latten_dim] is the backbone output size
        self.resize_1 = nn.Linear(emb_dim, latten_dim)
        self.new_encoder = nn.Sequential(self.encoder, self.resize_1)
        self.resize_2 = nn.Linear(latten_dim, emb_dim)
        patch_number = (image_size // patch_size) ** 2
        self.mlp = CrossAttention(in_channels1= int((1 - mask_ratio) * patch_number) + 1, in_channels2= patch_number + 1)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def image_patch(self, img):
        patches = self.encoder.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.encoder.pos_embedding
        patches, forward_indexes, backward_indexes = self.encoder.shuffle(patches)
        mask_patches = patches[: int(patches.size(0) * (1 - self.encoder.ratio))]
        return patches, mask_patches, forward_indexes, backward_indexes

    def forward(self, img):
        patches, mask_patches, forward_indexes, backward_indexes = self.image_patch(img)
        feature_patch = self.new_encoder(patches)
        feature_mask = self.new_encoder(mask_patches)
        # # # restore distillation features to original size
        patches = self.resize_2(feature_patch)
        mask_patches = self.resize_2(feature_mask)
        # # using Cross Attention combine original features to mask features
        out_features = self.mlp(mask_patches, patches)
        out_features = rearrange(out_features, 'n b f -> b n f')
        # # restore image according masked features
        predicted_img, mask = self.decoder(out_features, backward_indexes)
        # return predicted_img, mask
        return predicted_img, mask, feature_patch, feature_mask


class CrossAttention(nn.Module):
    def __init__(self, in_channels1, in_channels2):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels1, in_channels2 // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels2, in_channels2 // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels2, in_channels1, kernel_size=1)

    def forward(self, x1, x2):

        x1_unsquee = x1.unsqueeze(3)
        x2 = x2.unsqueeze(3)
        batch_size, channels, height, width = x1_unsquee.size()
        query = self.query_conv(x1_unsquee).view(batch_size, -1, height * width).permute(0, 2, 1)  # [B, HW, C']
        key = self.key_conv(x2).view(batch_size, -1, height * width)  # [B, C', HW]
        value = self.value_conv(x2).view(batch_size, -1, height * width)  # [B, C, HW]
        attn_weights = torch.bmm(query, key)  # [B, HW, HW]
        attn_weights = F.softmax(attn_weights, dim=2)
        attn_out = torch.bmm(value, attn_weights.permute(0, 2, 1))  # [B, C, HW]
        attn_out = attn_out.view(batch_size, channels, height * width)
        output = attn_out + x1
        return output

if __name__ == '__main__':
    shuffle = PatchShuffle(0.75)
    a = torch.rand(16, 2, 10)
    b, forward_indexes, backward_indexes = shuffle(a)
    print(b.shape)

    img = torch.rand(2, 3, 32, 32)
    encoder = MAE_Encoder()
    decoder = MAE_Decoder()
    features, backward_indexes = encoder(img)
    print(forward_indexes.shape)
    predicted_img, mask = decoder(features, backward_indexes)
    print(predicted_img.shape)
    loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)
    print(loss)