import torch
from thop import clever_format, profile
from scripts.model import Net

if __name__ == "__main__":
    device = torch.accelerator.current_accelerator()

    net = unet = Net(in_channels=4,
                     out_channels=4,
                     cond_emb_dim=256,
                     time_embed_dim=64,
                     include_spacing_input=False).to(device)

    dummy_input = torch.randn(1, 4, 128, 128, 128).to(device)
    timesteps = torch.rand(1).to(device)
    spacing_tensor = torch.rand(1,3).to(device)
    macs, params = profile(unet, inputs=(dummy_input,
                                         timesteps,
                                         spacing_tensor))

    # Format the numbers into a readable format (e.g., 4.14 GMac, 25.56 MParams)
    macs_readable, params_readable = clever_format([macs, params], "%.3f")
    print(f"Formatted MACs: {macs_readable}, Formatted Parameters: {params_readable}")
