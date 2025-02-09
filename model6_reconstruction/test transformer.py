import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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

        self.norm = nn.LayerNorm(dim)

    def forward(self, input):
        x, y = input  # HACK: for nn.Sequential
        B, N, E = x.shape
        _, M, _ = y.shape

        x = self.norm(x)  # (B, N, E)
        y = self.norm(y)  # (B, M, E)

        k_x = (
            self.w_k(x).view(B, -1, self.num_heads, E // self.num_heads).transpose(1, 2)
        )  # (B, num_heads, N, head_dim)
        v_x = (
            self.w_v(x).view(B, -1, self.num_heads, E // self.num_heads).transpose(1, 2)
        )  # (B, num_heads, N, head_dim)
        q_y = (
            self.w_q(y).view(B, -1, self.num_heads, E // self.num_heads).transpose(1, 2)
        )  # (B, num_heads, M, head_dim)

        attn_scores = torch.matmul(q_y, k_x.transpose(-2, -1)) * self.scale  # (B, num_heads, M, N)
        attn = F.softmax(attn_scores, dim=-1)  # (B, num_heads, M, N)
        out = attn @ v_x  # (B, num_heads, M, head_dim)
        out = rearrange(out, "B H M D -> B M (H D)")  # (B, M, E)

        out += y  # residual connection

        return out


# Initialize the model
model = MultiHeadAttention(seq_len_x=10, seq_len_y=10, dim=32, num_heads=8)

# Create example input tensors
x = torch.randn(2, 10, 32, requires_grad=True)  # (B, N, E)
y = torch.randn(2, 10, 32, requires_grad=True)  # (B, M, E)

# Perform a forward pass
output = model((x, y))

# Compute a dummy loss
loss = output.sum()

# Perform a backward pass
loss.backward()
print("x:", x)
print("y:", y)

# Check gradients
print("Gradient of w_q.weight:", model.w_q.weight.grad)
print("Gradient of w_k.weight:", model.w_k.weight.grad)
print("Gradient of w_v.weight:", model.w_v.weight.grad)

print(f"{model.w_q.weight.grad.norm()=}")
print(f"{model.w_k.weight.grad.norm()=}")
print(f"{model.w_v.weight.grad.norm()=}")

pass

# # Check if gradients are flowing through x and y
# print("Gradient of input x:", x.grad)
# print("Gradient of input y:", y.grad)
