import hashlib
import itertools
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
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

def view(ct_data, seg_data=None):
    blended = ct_data
    if seg_data is not None:
        colors = plt.get_cmap('tab10').colors
        for i, color in zip(range(seg_data.size(0)), itertools.cycle(colors)):
            blended = blend_images(
                image=blended, 
                label=seg_data[i, ...][None, ...], 
                alpha=0.5,
                cmap=ListedColormap(['none', color])
            )

    blended = blended.numpy().transpose(3, 1, 2, 0)
    z_max = blended.shape[0] - 1

    current_idx = 0

    fig, ax = plt.subplots()
    
    is_float = blended.max() <= 1.0
    im = ax.imshow(blended[current_idx], vmin=0, vmax=(1 if is_float else 255)) # Assuming 0-1 range based on your description
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')

    title = ax.text(
        0.02, 0.98, f'Slice {current_idx}', transform=ax.transAxes, 
        color='white', va='top', ha='left', fontsize=12, fontweight='bold'
    )

    def update_slice(step):
        nonlocal current_idx
        current_idx = np.clip(current_idx + step, 0, z_max)
        
        im.set_data(blended[current_idx])
        title.set_text(f'Slice {current_idx}')
        fig.canvas.draw_idle()

    # 4. Event Handlers
    def on_scroll(event):
        step = 1 if event.button == 'up' else -1
        update_slice(step)

    def on_key(event):
        if event.key in ['up', 'right']: update_slice(1)
        elif event.key in ['down', 'left']: update_slice(-1)

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

