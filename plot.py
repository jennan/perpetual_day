import cartopy.crs as ccrs
import xarray as xr
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import pandas as pd

def processing_data(target, pred):
    
    field = "channel_0003_scaled_radiance"
    
    target_da = target[field]
    target_da_new = xr.DataArray(
        target_da.data,
        dims=["latitude", "longitude"],  # dimension names
        coords={"latitude": target_da.latitude.data, "longitude": target_da.longitude.data},  # assign coords to dims
        name=field
    )
    
    pred_da = pred[field]
    pred_da_new = xr.DataArray(
        pred_da.data,
        dims=["latitude", "longitude"],  # dimension names
        coords={"latitude": pred_da.latitude.data, "longitude": pred_da.longitude.data},  # assign coords to dims
        name=field
    )
    diff_da = (pred - target)[field]
    diff_da_new = xr.DataArray(
        diff_da.data,
        dims=["latitude", "longitude"],  # dimension names
        coords={"latitude": diff_da.latitude.data, "longitude": diff_da.longitude.data},  # assign coords to dims
        name=field
    )
    data_time = target_da['time'].data
    return target_da_new, pred_da_new, diff_da_new, data_time

def plot_results(target, pred):

    target_da, pred_da, diff_da, data_time = processing_data(target, pred)

    proj = ccrs.PlateCarree()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 5),
                          subplot_kw={'projection': proj},
                           sharex=True,
                           sharey=True,
                           gridspec_kw={'wspace': 0.05}
                          )

    # Plot without colorbars
    im0 = target_da.plot.imshow(ax=axes[0], transform=proj, add_colorbar=False, vmin=0, vmax=1)
    im1 = pred_da.plot.imshow(ax=axes[1], transform=proj, add_colorbar=False, vmin=0, vmax=1)
    im2 = diff_da.plot.imshow(ax=axes[2], transform=proj, add_colorbar=False, vmin=-0.5, vmax=0.5, cmap="RdBu_r")
    # Add colorbar
    cbar = fig.colorbar(im2, ax=axes, orientation='vertical', fraction=0.05, pad=0.08, shrink=0.7)
    cbar.set_label("Diff")
    
    axes[0].set_title("Target")
    axes[1].set_title("Prediction")
    axes[2].set_title("Diff(Prediction-Target)")
    fig.suptitle(pd.to_datetime(data_time).strftime('%Y%m%d %H:%M'), fontsize=12, y=0.85)

    for ax in axes:
        ax.coastlines(resolution='10m', color='black', linewidth=1)
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.9, color='gray', alpha=0.9, linestyle='--')
        gl.xlocator = MultipleLocator(0.5)  # Longitude every 1 degree
        gl.ylocator = MultipleLocator(0.5)  # Latitude every 1 degree
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = False
        gl.bottom_labels = True
        if ax == axes[0]:
            gl.left_labels = True
        if ax == axes[-1]:
            gl.right_labels = True