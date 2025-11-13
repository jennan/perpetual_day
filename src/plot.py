import cartopy.crs as ccrs
import xarray as xr
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors as col


def normalizeList(listOld, new_min=0, new_max=1):
    old_min = min(listOld)
    old_range = max(listOld) - old_min
    new_min = new_min
    new_range = new_max - new_min
    output = [(n - old_min) / old_range * new_range + new_min for n in listOld]
    return output


def GetMICAPSIR1Dict(levfile):
    maplev = np.loadtxt(levfile, skiprows=0)
    # maplev = maplev[::-1]#array reverse()

    # normalize lev
    lev = maplev[0:, -1]
    levNew = normalizeList(lev, 0, 1)

    # get RBG
    cmap = maplev[0:, :-1]
    cmap = cmap / 255.0
    redList = cmap[0:, 0]
    greenList = cmap[0:, 1]
    blueList = cmap[0:, 2]

    redMatrix = [[0 for col in range(3)] for row in range(len(redList))]
    greenMatrix = [[0 for col in range(3)] for row in range(len(greenList))]
    blueMatrix = [[0 for col in range(3)] for row in range(len(blueList))]
    for i in range(len(redList)):
        redMatrix[i][0] = levNew[i]
        redMatrix[i][1] = redList[i]
        redMatrix[i][2] = redList[i]
        greenMatrix[i][0] = levNew[i]
        greenMatrix[i][1] = greenList[i]
        greenMatrix[i][2] = greenList[i]
        blueMatrix[i][0] = levNew[i]
        blueMatrix[i][1] = blueList[i]
        blueMatrix[i][2] = blueList[i]
    redMatrix.reverse()
    greenMatrix.reverse()
    blueMatrix.reverse()

    MICAPSIR1Dict = {"red": redMatrix, "green": greenMatrix, "blue": blueMatrix}
    return MICAPSIR1Dict


def processing_data(features, target, pred):
    feature_field = "channel_0013_brightness_temperature"
    field = "channel_0003_scaled_radiance"

    feature_da = features[feature_field]
    feature_da_new = xr.DataArray(
        feature_da.data - 273.15,
        dims=["latitude", "longitude"],  # dimension names
        coords={
            "latitude": feature_da.latitude.data,
            "longitude": feature_da.longitude.data,
        },
        name=field,
    )

    target_da = target[field]
    target_da_new = xr.DataArray(
        target_da.data,
        dims=["latitude", "longitude"],  # dimension names
        coords={
            "latitude": target_da.latitude.data,
            "longitude": target_da.longitude.data,
        },
        name=field,
    )

    pred_da = pred[field]
    pred_da_new = xr.DataArray(
        pred_da.data,
        dims=["latitude", "longitude"],  # dimension names
        coords={
            "latitude": pred_da.latitude.data,
            "longitude": pred_da.longitude.data,
        },
        name=field,
    )

    diff_da = (pred - target)[field]
    diff_da_new = xr.DataArray(
        diff_da.data,
        dims=["latitude", "longitude"],  # dimension names
        coords={
            "latitude": diff_da.latitude.data,
            "longitude": diff_da.longitude.data,
        },
        name=field,
    )

    data_time = target_da["time"].data

    return feature_da_new, target_da_new, pred_da_new, diff_da_new, data_time


def plot_results(features, target, pred):
    feature_da, target_da, pred_da, diff_da, data_time = processing_data(
        features, target, pred
    )

    proj = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(15, 5),
        subplot_kw={"projection": proj},
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.05},
    )

    my_cmapDict = GetMICAPSIR1Dict("./maplev_IR1_lyj.LEV")
    my_cmap = col.LinearSegmentedColormap("my_colormap", my_cmapDict, 256)

    # Plot without colorbars
    feature_da.plot.imshow(
        ax=axes[0], transform=proj, add_colorbar=False, vmin=-100, vmax=70, cmap=my_cmap
    )
    target_da.plot.imshow(
        ax=axes[1], transform=proj, add_colorbar=False, vmin=0, vmax=1, cmap="gray"
    )
    pred_da.plot.imshow(
        ax=axes[2], transform=proj, add_colorbar=False, vmin=0, vmax=1, cmap="gray"
    )
    im3 = diff_da.plot.imshow(
        ax=axes[3],
        transform=proj,
        add_colorbar=False,
        vmin=-0.5,
        vmax=0.5,
        cmap="RdBu_r",
    )

    # Add colorbar
    cbar = fig.colorbar(
        im3, ax=axes, orientation="vertical", fraction=0.05, pad=0.08, shrink=0.7
    )
    cbar.set_label("Diff")
    axes[0].set_title("Input(B13)")
    axes[1].set_title("Target(VIS)")
    axes[2].set_title("Prediction(VIS)")
    axes[3].set_title("Diff(Prediction-Target)")
    fig.suptitle(
        pd.to_datetime(data_time).strftime("%Y%m%d %H:%M"), fontsize=12, y=0.85
    )

    for ax in axes:
        ax.coastlines(resolution="10m", color="black", linewidth=1)
        # Add gridlines
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.9, color="gray", alpha=0.9, linestyle="--"
        )
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
