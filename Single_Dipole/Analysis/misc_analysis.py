import numpy as np
import matplotlib.pyplot as plt  

def get_err_hist(this_nn, this_lm):
    min_err = min(this_nn["loc_err"] + this_lm["loc_err"])
    max_err = max(this_nn["loc_err"] + this_lm["loc_err"])
    n_bins = int(round(np.sqrt(len(this_nn["loc_err"]))))
    hist_centers, hist_edges = get_hist_edges(n_bins=n_bins, min_edge=0, max_edge=max_err, min_offset=0, max_offset=1e-6)
    hist_nn, _ = np.histogram(this_nn["loc_err"], hist_edges)
    hist_lm, _ = np.histogram(this_lm["loc_err"], hist_edges)    
    return hist_centers, hist_nn, hist_lm

def get_hist_edges(data=None, n_bins=None, min_edge=None, max_edge=None, min_offset=None, max_offset=None):
    if n_bins is None:
        n_bins = int(round(np.sqrt(len(data))))
    n_bins = int(n_bins)
    if min_edge is None:
        min_edge = min(data)
    if max_edge is None:
        max_edge = max(data)
    if min_offset is None:
        min_offset = 0.001 * min_edge
    if max_offset is None:
        max_offset = min_offset
    hist_edges = np.linspace(min_edge-min_offset,max_edge+max_offset,n_bins+1)
    bin_width = np.abs(hist_edges[1] - hist_edges[0])
    hist_centers = np.linspace(hist_edges[0]+bin_width/2, hist_edges[-1]-bin_width/2, n_bins)
    return hist_centers, hist_edges
    
def get_means_and_stds_in_hist_edges(edges, indexing_data, data):
    means = np.zeros(len(edges)-1)
    stds = np.zeros(len(edges)-1)
    for i in range(len(edges)-1):
        selection = (edges[i+1] >= indexing_data) & (indexing_data >= edges[i] )
        means[i] = np.mean(data[selection])
        stds[i] = np.std(data[selection])    
    return means, stds
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    