import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_results(target, pred):

    field = "channel_0003_scaled_radiance"
    # Define projection
    proj = ccrs.PlateCarree()
    extent = [110, 155, -45, -10]

    # Create figure and axis
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), subplot_kw={'projection': proj})

    target[field].plot(ax=axes[0], transform=proj)
    pred[field].plot(ax=axes[1], transform=proj)
    (pred - target)[field].plot(ax=axes[2], transform=proj)
        
    # Add coastlines and gridlines to each subplot
    for ax in axes:
        ax.coastlines(resolution='110m', color='black', linewidth=1)
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
    
    plt.tight_layout()
    plt.title('Perperual Day')
    plt.show()

