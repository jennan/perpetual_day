import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot(ax, proj, extent, lons, lats, data, title):
    im = ax.pcolormesh(lons, lats, data, transform=proj, cmap='viridis')
    ax.set_title(title)

    # Set extent for Australia [lon_min, lon_max, lat_min, lat_max]
    ax.set_extent([110, 155, -45, -10], crs=proj)

    # Add coastlines
    ax.coastlines(resolution='10m')

    # Add land and ocean features
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    # Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(title)

def plot_results(sample, pred, lons, lats):
    
    # Define projection
    proj = ccrs.PlateCarree()
    extent = [110, 155, -45, -10]

    # Create figure and axis
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), subplot_kw={'projection': proj})

    sample = sample[0].cpu().detach().numpy()
    pred = pred[0].cpu().detach().numpy()
    
    lons_grid, lats_grid = np.meshgrid(lons,lats)
    
    # plot
    plot(axes[0], proj, extent, lons_grid, lats_grid, sample, "Ground Truth")
    plot(axes[1], proj, extent, lons_grid, lats_grid, pred, "Prediction")
    plot(axes[2], proj, extent, lons_grid, lats_grid, pred - sample, "Diff")
    
    plt.tight_layout()
    plt.title('Perperual Day')
    plt.show()

