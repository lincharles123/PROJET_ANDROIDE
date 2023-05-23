import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle

def load_archive(path):
    if isinstance(path, str):
        with open(path, 'rb') as f:
            archive = pickle.load(f)
    else:
        archive = path
    return pd.DataFrame(archive).values


def coverageMap2d(data, min_value, max_value, bin_size, plot=True):
    x = data[:, 0]
    y = data[:, 1]

    x = np.clip(x, min_value, max_value)
    y = np.clip(y, min_value, max_value)

    min_value_x = min_value
    max_value_x = max_value
    min_value_y = min_value
    max_value_y = max_value

    x_bins = bin_size
    y_bins = bin_size

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=(x_bins, y_bins),  range=((min_value, max_value), (min_value, max_value)))

    total_area = (max_value_x - min_value_x) * (max_value_y - min_value_y)
    covered_area = np.sum(heatmap > 0) * (total_area / heatmap.size)
    coverage = covered_area / total_area

    print(f'Coverage: {coverage * 100:.2f}%')

    if plot:
        plt.imshow(heatmap.T, extent=[min_value_x, max_value_x, min_value_y, max_value_y], origin='lower', cmap='viridis')
        plt.colorbar(label='Density')
        plt.title('Heatmap of the 2D data')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        
    return coverage


def coverageMap4d(data, min_value, max_value, bin_size=10, plot=True):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    c = data[:, 3]

    if plot:
        fig = plt.figure(figsize=(bin_size, bin_size))
    gs = gridspec.GridSpec(bin_size, bin_size)

    coverage = np.zeros((bin_size, bin_size))

    for i in range(bin_size):
        for j in range(bin_size):
            x_min = min_value + i * ((max_value - min_value) / bin_size)
            x_max = min_value + (i + 1) * ((max_value - min_value) / bin_size)
            y_min = min_value + j * ((max_value - min_value) / bin_size)
            y_max = min_value + (j + 1) * ((max_value - min_value) / bin_size)

            selected_z = z[(x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)]
            selected_c = c[(x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)]

            counts, _, _ = np.histogram2d(selected_z, selected_c, bins=bin_size)

            if np.sum(counts) > 0:
                coverage[j, i] = 1

            if plot:
                ax = plt.subplot(gs[j, i])
                ax.imshow(counts, cmap='viridis')
                ax.axis('off')
                
    coverage_ratio = np.sum(coverage) / (bin_size * bin_size)
    if plot:
        plt.title('coverage ratio: {:.2%}'.format(coverage_ratio))
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
    
    print("Coverage Ratio: {:.2%}".format(coverage_ratio))
    
    return coverage_ratio
    