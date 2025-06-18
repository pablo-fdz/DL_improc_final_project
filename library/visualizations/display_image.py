from matplotlib import pyplot as plt
import torch

def display_image(image:torch.Tensor, title:str='', cmap:str='gray', figsize=(4, 4)) -> None:
    image   = torch.einsum('dhw -> hwd', image)
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image, cmap=cmap)
    ax.set_title(title, fontsize=15)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    plt.close()