import hashlib
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from monai.visualize import blend_images


class DICOMDataAnomalyError(Exception):
    pass


def generate_uid(path):
    case_and_studey = '/'.join(str(path).split('/')[-2:])
    # 16 bit UID ~ 1 in 14 billion chanche for collision for 1 million datapoints
    uid = hashlib.shake_256(case_and_studey.encode()).hexdigest(8)
    return uid

def visualize(ct_data, seg_data, slice_idx=None, save_file=None): 
    if slice_idx is None:
        slice_idx = ct_data.shape[-1] // 2
    blended = ct_data[..., slice_idx]
    colors = plt.get_cmap('tab10').colors
    for i, color in zip(range(seg_data.size(0)), itertools.cycle(colors)):
        blended = blend_images(
            image=blended, 
            label=seg_data[i, ..., slice_idx][None, ...], 
            alpha=0.5,
            cmap=ListedColormap(['none', color])
        )
    plt.figure(figsize=(10, 10))
    plt.imshow(blended.permute(1, 2, 0)) 
    plt.axis('off')
    plt.title(f'Overlay at Slice {slice_idx}')
    if save_file is not None:
        plt.savefig(save_file)
    else:
        plt.show()

