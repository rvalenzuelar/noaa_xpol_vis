# -*- coding: utf-8 -*-
"""
  Created on Wed Jun 22 15:43:52 2016
  
  Source: Jon Kington
  http://stackoverflow.com/questions/3798333/
  image-information-along-a-polar-coordinate-system

  This module uses scipy.map_coordinates which seems
  to work in pixel coordinates. Perhaps for physical
  coordinates (e.g. grids in km, degrees) kdTree is
  better


  Raul Valenzuela
  raul.valenzuela@colorado.edu

"""

import numpy as np
import scipy.ndimage as ndimage


def project_into_polar(data,origin=None,order=1):
    """
      Projects a 2D numpy array ("data") into a polar coordinate system.
      "origin" is a tuple of (x0, y0) and defaults to the center
      of array.
    """
    ny, nx = data.shape

    ' Determine r and theta '
    x, y = index_coords(data, origin=origin)
    r, _ = cart2polar(x, y)


    ' Grid in polar space '
    r_i = np.linspace(r.min(), r.max(), ny)   
#    theta_i = np.linspace(0.66*np.pi,1.83*np.pi, nx)    
    theta_i = np.linspace(0,2*np.pi, nx)    
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    ' Project the r and theta grid back into pixel coordinates '
    xi, yi = polar2cart(r_grid, theta_grid)
    xi += origin[0] # grid relative to origin
    yi += origin[1] 
    xi, yi = xi.flatten(), yi.flatten()
    coords = np.vstack((xi, yi)) # (map_coordinates requires a 2xn array)

    ' interpolated array '
    zi = ndimage.map_coordinates(data, coords, order=order)
    zi = np.reshape(zi,(ny, nx))
  
    return zi, theta_i, r_i

def project_into_cart(data, theta, r, origin=None, order=1):
    """
      Projects a 2D numpy array ("data") into a cartesian coordinate system.
    """
    ny, nx = data.shape

    theta_grid, r_grid = np.meshgrid(theta,r)
    
#    ' Input grid in polar space '
#    r_i = np.linspace(0, 180, nx)   
#    theta_i = np.linspace(0,2*np.pi, ny)    
#    theta_grid, r_grid = np.meshgrid(theta_i, r_i)
#
    ' Project the r and theta grid back into pixel coordinates '
    xi, yi = polar2cart(r_grid, theta_grid)
        
    xi += origin[0] # grid relative to origin
    yi += origin[1] 
    xi, yi = xi.flatten(), yi.flatten()
    coords = np.vstack((xi, yi)) # (map_coordinates requires a 2xn array)

    ' interpolated array '
    zi = ndimage.map_coordinates(data, coords, order=order)
    zi = np.reshape(zi,(ny,nx))
  
    return zi, xi, yi

  
def index_coords(data, origin=None):
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image."""
    ny, nx = data.shape
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y

def cart2polar(x, y):
     
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def polar2cart(r, theta):
  
    x = r * np.cos(theta)
    y = r * np.sin(theta)*-1
    return x, y 
  
  
  