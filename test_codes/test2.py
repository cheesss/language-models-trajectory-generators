import torch

def _generate_color_palette(num_objects):
    palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
    return [tuple((i * palette) % 255) for i in range(num_objects)]


print(len(_generate_color_palette(2)))
