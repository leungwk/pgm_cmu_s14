import numpy as np
import scipy.io
import pandas as pd
from collections import Counter
# from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift

data_dir = 'data/hw1/'
img_dir = 'img/'

def _load_images(path):
    mat_file = scipy.io.loadmat(path)
    img_orig = mat_file['origImg'] # z
    img_noisy = mat_file['noisyImg'] # x
    return img_orig, img_noisy

def edit_distance(img_base, img_pred):
    return np.sum(img_base != img_pred) # assuming binary # use np.sum() otherwise sum() will return an array rather number
    # return sum(np.abs(img_pred -img_base))/2. # /2. because {-1,+1}

#### ================================================================
#### local, vote filters
#### ================================================================

def _cross_1(coord_in, shape):
    sx, sy = shape
    def oob(coord):
        cx,cy = coord
        return not ((0 <= cx < sx) and (0 <= cy < sy))
    cx,cy = coord_in
    return [c for c in [(cx+1,cy), (cx,cy-1), (cx-1,cy), (cx,cy+1)] if not oob(c)]


def _box_1(coord_in, shape):
    sx, sy = shape
    def oob(coord):
        cx,cy = coord
        return not ((0 <= cx < sx) and (0 <= cy < sy))
    cx,cy = coord_in
    return [c for c in [(cx+1,cy), (cx+1,cy-1), (cx,cy-1), (cx-1,cy-1), (cx-1,cy), (cx-1,cy+1), (cx,cy+1), (cx+1,cy+1)] if not oob(c)]


def _box_2(coord_in, shape):
    sx, sy = shape
    def oob(coord):
        cx,cy = coord
        return not ((0 <= cx < sx) and (0 <= cy < sy))
    cx,cy = coord_in
    return [c for c in
            [(cx+1,cy), (cx+1,cy-1), (cx,cy-1), (cx-1,cy-1), (cx-1,cy), (cx-1,cy+1), (cx,cy+1), (cx+1,cy+1)] +
            [(cx+2,cy), (cx+2,cy-1), (cx+2,cy-2), (cx+1,cy-2), (cx,cy-2), (cx-1,cy-2), (cx-2,cy-2), (cx-2,cy-1), (cx-2,cy), (cx-2,cy+1), (cx-2,cy+2), (cx-1,cy+2), (cx,cy+2), (cx+1,cy+2), (cx+2,cy+2), (cx+2,cy+1)] if not oob(c)]


def _diamond_2(coord_in, shape):
    sx, sy = shape
    def oob(coord):
        cx,cy = coord
        return not ((0 <= cx < sx) and (0 <= cy < sy))
    cx,cy = coord_in
    return [c for c in
            [(cx+1,cy), (cx,cy-1), (cx-1,cy), (cx,cy+1)] +
            [(cx+2,cy), (cx+1,cy-1), (cx,cy-2), (cx-1,cy-1), (cx-2,cy), (cx-1,cy+1), (cx,cy+2), (cx+1,cy+1)] if not oob(c)]


def _vote_filter(img_in, neighbour_f):
    img_new = np.zeros(img_in.shape)
    for coord, _ in np.ndenumerate(img_in):
        cnt = Counter([img_in[c] for c in neighbour_f(coord, img_in.shape)])
        val, _ = cnt.most_common()[0]
        img_new[coord] = val
    return img_new.astype(np.int16)


def _thres_filter(img_in, neighbour_f, p_thres):
    """Like _vote_filter, but allow the threshold for a pixel change to be set"""
    img_new = np.zeros(img_in.shape)
    for coord, val_old in np.ndenumerate(img_in):
        res = [img_in[c] for c in neighbour_f(coord, img_in.shape)]
        n_neighbours = 1.0*len(res)
        cnt = Counter(res)
        new_val = 1 if val_old == -1 else -1 # assuming binary {-1,+1}
        if cnt.get(new_val, 0)/n_neighbours >= p_thres:
            img_new[coord] = new_val
        else:
            img_new[coord] = val_old
    return img_new.astype(np.int16)

#### ================================================================
#### energy-based
#### ================================================================

def energy_at(coord_in, z, x, h, beta, nu):
    """As opposed to the full energy function, calculate only the terms affected by the input z.
"""
    sx, sy = z.shape
    def oob(coord):
        cx,cy = coord
        return not ((0 <= cx < sx) and (0 <= cy < sy))
    cx, cy = coord_in
    coords = [c for c in [(cx+1,cy), (cx,cy-1), (cx-1,cy), (cx,cy+1)] if not oob(c)]
    term1 = h*z[coord_in]
    term2 = beta*sum([z[coord_in]*z[c] for c in coords])
    term3 = z[coord_in]*x[coord_in]
    return term1 -term2 -term3


def energy_denoise(img_noisy_in, h, beta, nu):
    img_denoise_energy = img_noisy_in.copy()
    prev_energy = 0
    for coord, old_pixel in np.ndenumerate(img_denoise_energy):
        old_energy_at = energy_at(coord, img_denoise_energy, img_noisy_in, h, beta, nu)
        old_energy = prev_energy +old_energy_at
        new_pixel = 1 if old_pixel == -1 else -1 # assuming binary {-1,+1}
        img_denoise_energy[coord] = new_pixel
        new_energy_at = energy_at(coord, img_denoise_energy, img_noisy_in, h, beta, nu)
        new_energy = prev_energy +new_energy_at
        if new_energy < old_energy:
            prev_energy = new_energy
        else:
            img_denoise_energy[coord] = old_pixel
    return img_denoise_energy

#### ================================================================
####
#### ================================================================

def _calc_error(img_base, imgs):
    total_pixels = img_base.shape[0]*img_base.shape[1]
    acc_err = []
    for key, img in imgs.items():
        err = edit_distance(img_base, img)
        acc_err.append((key, err))
    df_err = pd.DataFrame(acc_err, columns=['title', 'err']).set_index('title')
    df_err['p'] = 1.*df_err['err']/total_pixels
    df_err.sort_values('p', inplace=True)
    return df_err

def _plot_images(img_base, imgs, params_energy, output_path):
    display_first = [('orig',), ('noisy',)]
    df_err = _calc_error(img_orig, imgs)
    total_pixels = img_base.shape[0]*img_base.shape[1]
    img_display_order = display_first +[k for k in df_err.index if k not in display_first]

    plt.close()
    plt.ion()
    shape_subplots = (3,5)
    fig, axs = plt.subplots(shape_subplots[0],shape_subplots[1],figsize=(22,12),dpi=600)
    img_display_order += [None]*(shape_subplots[0]*shape_subplots[1] -len(img_display_order)) # pad
    fig.suptitle("""Image denoising schemes and error""")
    fig.tight_layout()
    plt.figtext(0.02,0.00, r"""
    Pixel switch threshold for vote filters is 0.5 unless otherwise indicated.
    energy(h={h}, $\beta$={beta}, $\nu$={nu})
    error = edit_distance(orig, predicted)
    """.format(**params_energy), fontsize=10)
    for key, (_, ax) in zip(img_display_order, np.ndenumerate(axs)):
        if key is None:
            ax.axis('off')
            continue
        img = imgs[key]
        err = edit_distance(img_base, img)
        title = ', '.join(key)
        ax.set_title("{}\n{:.2f}%".format(title, 1.*err/total_pixels*100), fontsize=10)
        ax.imshow(img, cmap='Greys', interpolation='nearest')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    plt.savefig(output_path, format='png', bbox_inches='tight')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input-mat', help='input file', default=data_dir +'hw1_images.mat')
    parser.add_argument('--output-img', help='output img', default=img_dir +'hw01_s4_-_image_denoising.png')
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.set_defaults(plot=False)
    args = parser.parse_args()

    img_orig, img_noisy = _load_images(args.input_mat)
    total_pixels = img_orig.shape[0]*img_orig.shape[1]

    #### denoise

    imgs = {}
    imgs[('orig',)] = img_orig
    imgs[('noisy',)] = img_noisy

    imgs[('noisy', 'cross-1')] = _vote_filter(img_noisy, _cross_1)
    imgs[('noisy', 'cross-1', 'cross-1')] = _vote_filter(_vote_filter(img_noisy, _cross_1), _cross_1)
    imgs[('noisy', 'box-1')] = _vote_filter(img_noisy, _box_1)
    imgs[('noisy', 'cross-1', 'box-1')] = _vote_filter(_vote_filter(img_noisy, _cross_1), _box_1)
    imgs[('noisy', 'box-2')] = _vote_filter(img_noisy, _box_2)
    imgs[('noisy', 'diamond-2')] = _vote_filter(img_noisy, _diamond_2)
    imgs[('noisy', 'box-2-18/24')] = _thres_filter(img_noisy, _box_2, 18/24.)

    params_energy = {
        'h': 0.0,
        'beta': 2.0,
        'nu': 1.0,
        }
    img_denoise_energy = energy_denoise(img_noisy, **params_energy)
    imgs[('noisy', 'energy')] = img_denoise_energy
    imgs[('noisy', 'energy', 'cross-1')] = _vote_filter(img_denoise_energy, _cross_1)
    imgs[('noisy', 'energy', 'box-1')] = _vote_filter(img_denoise_energy, _box_1)
    imgs[('noisy', 'energy', 'box-2')] = _vote_filter(img_denoise_energy, _box_2)
    imgs[('noisy', 'energy', 'diamond-2')] = _vote_filter(img_denoise_energy, _diamond_2)
    imgs[('noisy', 'energy', 'box-2-18/24')] = _thres_filter(img_denoise_energy, _box_2, 18/24.)

    #### stats and plot

    df_err = _calc_error(img_orig, imgs)
    if args.plot:
        _plot_images(img_orig, imgs, params_energy, args.output_img)

    print(df_err)
