import torch
from torchviz import make_dot
from modules.transformer import Transformer  # Ensure to import your model correctly

# Assuming the Transformer class is properly defined and imported
model = Transformer(n_layer=1, seq_len_x=10, seq_len_y=10, dim=512, num_heads=8)

# Create dummy inputs according to your model's requirements
x = torch.randn(1, 10, 512)  # Batch size of 1, sequence length of 10, dimension of 512
y = torch.randn(1, 10, 512)  # Adjust dimensions as necessary

# Forward pass through the model to get the output
output = model((x, y))

# Use torchviz to create a dot graph of the model
dot = make_dot(output, params=dict(list(model.named_parameters()) + [("x", x), ("y", y)]))

# Render the graph
dot.render("Transformer_Computation_Graph", format="png")  # Saves the graph as a PNG image
