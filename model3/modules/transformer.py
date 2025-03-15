# transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig

from .classifier import Classifier
from .transformer_components import CustomTransformerEncoderLayer, CustomTransformerEncoder
from .utils import preprocess_data


class Encoder(nn.Module):
    def __init__(
        self, seq_len_x: int, seq_len_y: int, dim: int, num_heads: int = 8, qkv_bias: bool = False, qk_scale=None
    ):
        super(Encoder, self).__init__()
        self.s_attn = MultiHeadAttention(seq_len_x, seq_len_y, dim, num_heads, qkv_bias, qk_scale)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=dim * 4)

    def forward(self, input):
        """
        Args:
            x: (B, N, E)
        """
        x, _, _ = input  # HACK: for nn.Sequential
        attn_output, attn_weights = self.s_attn((x, x))
        x = x + attn_output
        x = x + self.feed_forward(x)
        return x, x, attn_weights


class Decoder(nn.Module):
    def __init__(
        self, seq_len_x: int, seq_len_y: int, dim: int, num_heads: int = 8, qkv_bias: bool = False, qk_scale=None
    ):
        super(Decoder, self).__init__()
        self.s_attn = MultiHeadAttention(seq_len_x, seq_len_y, dim, num_heads, qkv_bias, qk_scale)
        self.c_attn = MultiHeadAttention(seq_len_x, seq_len_y, dim, num_heads, qkv_bias, qk_scale)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=dim * 4)

    def forward(self, input):
        """
        Args:
            x: (B, N, E) 3d latent vector
            y: (B, M, E) 2d latent vector
        """
        x, y, _, _ = input  # HACK: for nn.Sequential
        y_attn_output, s_attn_weights = self.s_attn((y, y))
        y = y + y_attn_output
        y_attn_output, c_attn_weights = self.c_attn((x, y))
        y = y + y_attn_output 
        y = y + self.feed_forward(y)
        
        # y = y + self.s_attn((y, y))
        # y = y + self.c_attn((x, y))
        # y = y + self.feed_forward(y)
        return x, y, s_attn_weights, c_attn_weights


class Transformer(nn.Module):
    def __init__(self, n_layer: int, seq_len_x, seq_len_y, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(Transformer, self).__init__()
        self.encoder = nn.Sequential(
            *[Encoder(seq_len_x, seq_len_x, dim, num_heads, qkv_bias, qk_scale) for _ in range(n_layer)]
        )
        self.decoder = nn.Sequential(
            *[Decoder(seq_len_x, seq_len_y, dim, num_heads, qkv_bias, qk_scale) for _ in range(n_layer)]
        )

    def forward(self, input):
        """
        Args:
            x: (B, N, E) 3d embedding
            y: (B, M, E) 2d embedding
        """
        x1, y1 = input  # HACK: for nn.Sequential 

        # x2, _ = self.encoder((x1, x1))
        x2, _, enc_attn_weights = self.encoder((x1, x1, x1))

        # x3, y2 = self.decoder((x2, y1))
        x3, y2, s_attn_weights, c_attn_weights = self.decoder((x2, y1, x2, y1))

        return x3, y2, enc_attn_weights, s_attn_weights, c_attn_weights  # x: latent vector of 3d, y: latent vector of 2d which we should focus


class MultiHeadAttention(nn.Module):
    def __init__(self, seq_len_x, seq_len_y, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.w_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.norm_x = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)
        
        self.proj = nn.Linear(dim ,dim) 

    def forward(self, input):
        """
        Args:
            x: (B, N, E)
            y: (B, M, E)
            x: key, value y: query
            x -> y
        """
        x, y = input  # HACK: for nn.Sequential
        B, N, E = x.shape
        _, M, _ = y.shape

        x = self.norm_x(x)  # (B, N, E)
        if id(x) == id(y):  # self attention
            y = self.norm_x(y)  # (B, M, E)
        else:  # cross attention
            y = self.norm_y(y)

        k_x = (
            self.w_k(x).reshape(B, -1, self.num_heads, E // self.num_heads).transpose(1, 2)
        )  # (B, num_heads, M, head_dim)
        v_x = (
            self.w_v(x).reshape(B, -1, self.num_heads, E // self.num_heads).transpose(1, 2)
        )  # (B, num_heads, M, head_dim)
        q_y = (
            self.w_q(y).reshape(B, -1, self.num_heads, E // self.num_heads).transpose(1, 2)
        )  # (B, num_heads, M, head_dim)

        attn_scores = torch.matmul(q_y, k_x.transpose(-2, -1)) * self.scale  # (B, num_heads, M, N)
        attn = F.softmax(attn_scores, dim=-1)  # (B, num_heads, M, N)
        out = attn @ v_x  # (B, num_heads, M, head_dim)
        out = rearrange(out, "B H M D -> B M (H D)")  # (B, M, E)
        
        out = self.proj(out)
        
        return out, attn


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, seq_len_x, seq_len_y, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(MultiHeadCrossAttention, self).__init__()

        self.num_heads = num_heads

        self.seq_len_x = seq_len_x
        self.seq_len_y = seq_len_y

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.w_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, y):
        """
        Args:
            x: (B, N, E)
            y: (B, M, E)
            x: key, value y: query
            x -> y
        """
        B, N, E = x.shape
        _, M, _ = y.shape

        x = self.norm(x)  # (B, N, E)
        y = self.norm(y)  # (B, M, E)

        k_x = (
            self.w_k(x).view(B, -1, self.num_heads, E // self.num_heads).transpose(1, 2)
        )  # (B, num_heads, M, head_dim)
        v_x = (
            self.w_v(x).view(B, -1, self.num_heads, E // self.num_heads).transpose(1, 2)
        )  # (B, num_heads, M, head_dim)
        q_y = (
            self.w_q(y).view(B, -1, self.num_heads, E // self.num_heads).transpose(1, 2)
        )  # (B, num_heads, M, head_dim)

        attn_scores = torch.matmul(q_y, k_x.transpose(-2, -1)) * self.scale  # (B, num_heads, M, N)
        attn = F.softmax(attn_scores, dim=-1)  # (B, num_heads, M, N)
        out = attn @ v_x  # (B, num_heads, M, head_dim)
        out = rearrange(out, "B H M D -> B M (H D)")  # (B, M, E)

        out += y  # residual connection

        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, seq_len_x, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(MultiHeadSelfAttention, self).__init__()

        self.num_heads = num_heads

        self.seq_len_x = seq_len_x

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.w_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: (B, N, E)
            y: (B, M, E)
            x: key, value y: query
            x -> y
        """
        B, N, E = x.shape
        _, M, _ = x.shape

        x = self.norm(x)  # (B, N, E)

        k_x = (
            self.w_k(x).view(B, -1, self.num_heads, E // self.num_heads).transpose(1, 2)
        )  # (B, num_heads, M, head_dim)
        v_x = (
            self.w_v(x).view(B, -1, self.num_heads, E // self.num_heads).transpose(1, 2)
        )  # (B, num_heads, M, head_dim)
        q_x = (
            self.w_q(x).view(B, -1, self.num_heads, E // self.num_heads).transpose(1, 2)
        )  # (B, num_heads, M, head_dim)

        attn_scores = torch.matmul(q_x, k_x.transpose(-2, -1)) * self.scale  # (B, num_heads, M, N)
        attn = F.softmax(attn_scores, dim=-1)  # (B, num_heads, M, N)
        out = attn @ v_x  # (B, num_heads, M, head_dim)
        out = rearrange(out, "B H M D -> B M (H D)")  # (B, M, E)

        out += x  # residual connection

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.net(self.norm(x))


class PatchEmbed3D(nn.Module):
    """Patch Embedding of CT images

    Args:
        img_size (tuple): Size of the input image
        patch_size (tuple): Size of the patch
        in_channels (int): Number of input channels (Default: 1)
        embed_dim (int): Dimension of the embedding
        norm_layer (nn.Module, optional): Normalization layer (Default: None)
    """

    def __init__(self, img_size=(512, 512, 64), patch_size=(16, 16, 4), in_chans=1, embed_dim=256, norm_layer=None):
        super(PatchEmbed3D, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        assert img_size[0] % patch_size[0] == 0, "Image dimensions must be divisible by the patch size"
        assert img_size[1] % patch_size[1] == 0, "Image dimensions must be divisible by the patch size"
        assert img_size[2] % patch_size[2] == 0, "Image dimensions must be divisible by the patch size"

        patches_resolution = [img_size[i] // patch_size[i] for i in range(3)]
        self.patches_resolution = patches_resolution

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )  # Linear embedding outputs 3D tensor

        if norm_layer:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """
        Input: (B, C, H, W, D)
            patch size: (P, P, P)
            outputs: (B, C//P * H//P * W//P, E)

        Output: (B, E, H//P, W//P, D//P)
        """
        x = self.proj(x)  # (B, C, H, W, D) -> (B, embed_dim, H//P[0], W//P[1], D//P[2])
        if self.norm:
            x = self.norm(x)
        return x


class PatchEmbed2D(nn.Module):
    """Patch Embedding of CT images

    Args:
        img_size (tuple): Size of the input image
        patch_size (tuple): Size of the patch
        in_channels (int): Number of input channels (Default: 1)
        embed_dim (int): Dimension of the embedding
        norm_layer (nn.Module, optional): Normalization layer (Default: None)
    """

    def __init__(self, img_size=(1024, 1024), patch_size=(32, 32), in_chans=1, embed_dim=256, norm_layer=None):
        super(PatchEmbed2D, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        assert img_size[0] % patch_size[0] == 0, "Image dimensions must be divisible by the patch size"
        assert img_size[1] % patch_size[1] == 0, "Image dimensions must be divisible by the patch size"

        patches_resolution = [img_size[i] // patch_size[i] for i in range(2)]
        self.patches_resolution = patches_resolution

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )  # Linear embedding outputs 2D tensor

        if norm_layer:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """
        Input: (B, C, H, W, D)
            patch size: (P, P, P)
            outputs: (B, C//P * H//P * W//P, E)

        Output: (B, E, H//P, W//P, D//P)
        """
        x = self.proj(x)  # (B, C, H, W) -> (B, embed_dim, H//P[0], W//P[1])
        if self.norm:
            x = self.norm(x)
        return x


class TransformerModel(nn.Module): 
    def __init__(
        self,
        hydra_config: DictConfig,  # hydra config
        model_config,  # model config
        data_example: dict,  # for model configuration depending on the data size and etc
    ):
        super(TransformerModel, self).__init__()
        self.base_path = hydra_config.data.data_dir
        self.patch_embed3d = PatchEmbed3D(patch_size=model_config.n_patch3d, embed_dim=model_config.n_embed)
        self.patch_embed2d = PatchEmbed2D(patch_size=model_config.n_patch2d, embed_dim=model_config.n_embed)

        self.num_heads = model_config.n_head
        # to configure the data size and types
        data = preprocess_data(self.base_path, data_example)

        B, _, H, W, D = data["CT_image"].shape
        seq_len_x = (
            (H // model_config.n_patch3d[0]) * (W // model_config.n_patch3d[1]) * (D // model_config.n_patch3d[2])
        )

        B, _, H, W = data["img"].shape
        seq_len_y = (H // model_config.n_patch2d[0]) * (W // model_config.n_patch2d[1])

        self.pe_x = nn.Parameter(torch.randn(1, seq_len_x, model_config.n_embed) * 1e-3)
        self.pe_y = nn.Parameter(torch.randn(1, seq_len_y, model_config.n_embed) * 1e-3)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_config.n_embed) * 1e-3)
        
        self.proj3d = nn.Linear(model_config.n_embed, model_config.n_embed)
        self.proj2d = nn.Linear(model_config.n_embed, model_config.n_embed)
        
        encoder_layer = CustomTransformerEncoderLayer(  
            d_model=model_config.n_embed,
            nhead=model_config.n_head,
            dim_feedforward=2048,
            dropout=0.1,
        )
        
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_layers=model_config.n_layer)
        self.classifier = Classifier(dim=model_config.n_embed, hidden_dim=2048)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        patches_3d = self.patch_embed3d(x)  # B E H W D (Batch, Embedding, Height, Width, Depth)
        patches_3d = rearrange(patches_3d, "B E H W D -> B (H W D) E")
        patches_3d = self.proj3d(patches_3d)  # 3d cubelet patch embedding
        patches_3d += self.pe_x

        patches_2d = self.patch_embed2d(y[:, 0:1])  # B E H W (Batch, Embedding, Height, Width)
        patches_2d = rearrange(patches_2d, "B E H W -> B (H W) E")
        patches_2d = self.proj2d(patches_2d)  # 2d square patch embedding
        patches_2d += self.pe_y
        
        features = torch.cat((self.cls_token, patches_3d, patches_2d), dim=1)
        
        out = self.transformer_encoder(features)
        
        logit = self.classifier(out[0][:, 0, :])
        return logit, out[1]
