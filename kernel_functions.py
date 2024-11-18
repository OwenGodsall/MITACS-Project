# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:04:29 2024

@author: ogods
"""

import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator



def global_mean(lat, data):
    lon_mean = np.mean(data, axis=1)
    lat_rad = np.radians(lat)
    weights = np.cos(lat_rad)
    weight_mean = np.average(lon_mean, weights=weights, axis=0)
    return weight_mean

def plot_contour_map(x, y, z, title, units="", vmin = None,vmax = None):
    
        
    if vmax == None:
        vmax = np.max(z)
        vmin = np.min(z)
        v = np.max([abs(vmin), abs(vmax)])
        vmax = v
        vmin = -v
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
    
    contour = ax.contourf(x, y, z, transform=ccrs.PlateCarree(), cmap=plt.cm.seismic, levels=20, vmin = vmin, vmax = vmax)
    
    ax.coastlines()
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False
    
    cbar = plt.colorbar(contour, ax=ax, shrink=0.7, orientation='horizontal')
    cbar.set_label(f'{units}')
    
    plt.title(title)
    
    plt.show()
    



def coarsen(lat, lon, new_lat, new_lon, data):
    # Create mesh grids for the original and new coordinates
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    new_lon_grid, new_lat_grid = np.meshgrid(new_lon, new_lat)

    # Flatten the grid coordinates and data
    points = np.array([lat_grid.flatten(), lon_grid.flatten()]).T
    values = data.flatten()
    
    new_points = np.array([new_lat_grid.flatten(), new_lon_grid.flatten()]).T
    
    new_values = griddata(points, values, new_points, method='linear')

    
    # Reshape the interpolated data to match the new grid shape
    new_values = new_values.reshape((len(new_lat), len(new_lon)))
    
    return new_values



def cross_section(x, y, z,  title, units):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    
    
    contour = ax.contourf(x, y, z, cmap=plt.cm.seismic, levels=20)
    
    
    cbar = plt.colorbar(contour, ax=ax, shrink=1)
    cbar.set_label(units, fontsize = 15)
    cbar.ax.tick_params(labelsize = 15)
    
    ax.set_ylim(1000, 0)  # Ensure y-axis displays from top (1000 hPa) to bottom (0 hPa)
    ax.set_yticks(np.linspace(0, 1000, 11))  # Adjust ticks as needed
    ax.set_ylabel('hPa', fontsize = 15)
    
    
    ax.set_xticks(np.linspace(-90, 90, 7))
    ax.set_xlabel('Latitude', fontsize = 15)
    
    
    ax.set_aspect('auto')
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    
    plt.title(title, fontsize = 20)
    
    
    plt.show()



def interpolate_3d(x, y, z, values, new_x, new_y, new_z):

    # Create an interpolator for the grid
    interpolator = RegularGridInterpolator(
        (x, y, z), values, bounds_error=False, fill_value=np.nan
    )

    # Create a meshgrid for the new (new_x, new_y, new_z) points
    new_X, new_Y, new_Z = np.meshgrid(new_x, new_y, new_z, indexing='ij')

    # Flatten the meshgrid points into a list of 3D points
    interp_points = np.array([new_X.ravel(), new_Y.ravel(), new_Z.ravel()]).T

    # Perform the interpolation
    interpolated_values = interpolator(interp_points)

    # Reshape the interpolated values to match the shape of the meshgrid
    interpolated_values = interpolated_values.reshape(new_X.shape)

    # If a mask is provided, set the masked regions to zero
    
    return interpolated_values





def unpack_file(ddir, names=None, mode = 'model'):
    model_datasets = {}
    
    
    if mode == 'model':
        
        for file in os.listdir(ddir):
            
            if names==None:
                
                f = netCDF4.Dataset(os.path.join(ddir,file))
                
             
                keys_list = list(f.variables.keys())
                
                var_name = keys_list[-1]
                #print(keys_list)
                long_name = f.variables[f'{var_name}'].long_name
                print(f'{var_name} - {long_name}' )
                
                print(keys_list)
                model_datasets[f'{var_name}'] = f.variables[f'{var_name}'][:]
                
            else:
                variables = file.split('_')
                for element in variables:
                    
                    if element in names:
                        
                        var_name = element
                    
                        f = netCDF4.Dataset(os.path.join(ddir,file))
                        
                     
                        keys_list = list(f.variables.keys())
                        #print(keys_list)
                        long_name = f.variables[f'{var_name}'].long_name
                        print(f'{var_name} - {long_name}' )
                        model_datasets[f'{var_name}'] = f.variables[f'{var_name}'][:]
            
            
        if model_datasets[f'{var_name}'].ndim == 3:
        
            lat, lon = f.variables['lat'][:], f.variables['lon'][:]
        
            return model_datasets, lat, lon
        
        else:
            
            lat, lon, levels = f.variables['lat'][:], f.variables['lon'][:], f.variables['plev'][:]
        
            return model_datasets, lat, lon, levels
            
    if mode == 'kernel':
        for file in os.listdir(ddir):
            
            variables = file.split('_')
            if any(element in names for element in variables):
                
                f = netCDF4.Dataset(os.path.join(ddir,file))
                
             
                keys_list = list(f.variables.keys())
                
                print(keys_list)
                
                model_datasets = f.variables
            
            
        
        lat, lon = f.variables['latitude'][:], f.variables['longitude'][:]
        
        
        
        return model_datasets, lat, lon




def q_units_transform(dq, ta, q):
    Rv = 487.5
    Lv = 2.5e6
    
    ta_sq = np.square(ta)
    
    
    new_units = np.multiply(dq,ta_sq)*Rv/Lv
    
    return new_units



















            
    