# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:24:41 2016

@author: raul
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import xpol_tta_analysis as xta
from scipy.spatial import cKDTree

class edit_polar:
  
    def __init__(self,array=None,origin=None):
      
      self.array_in = array.copy()
      self.array = array.copy()
      self.origin = origin

      if array is not None and origin is not None:
          self.edit()


    def remove_line(self,coords_domain,az=None,n=None):
        
        maxrange = 200
        direction = az * (np.pi/180.)
        
        r = np.arange(maxrange)
        theta = np.array([direction]*maxrange)
    
        xi,yi=polar2cart(r, theta)
        xi+=self.origin[0]
        yi+=self.origin[1]
        coords_i = zip(xi,yi)  
        
        ''' create kdTree with entire domain'''
        tree = cKDTree(coords_domain)
    
        neigh = n
        dist, idx = tree.query( coords_i, 
                                k=neigh,
                                eps=0,
                                p=1,
                                distance_upper_bound=10)
    
        idx = idx.flatten()
        for i in idx:
          try:
            self.array[coords_domain[i]]=np.nan
          except IndexError:
            pass
        
    def remove_ring(self,coords_domain,rang=None,n=None):
    
        
        theta = np.linspace(0,2*np.pi,300)
        r = np.array([rang]*theta.size)
        
        xi,yi=polar2cart(r, theta)
        xi+=self.origin[0]
        yi+=self.origin[1]
        coords_i = zip(xi,yi)  

        ''' create kdTree with entire domain'''
        tree = cKDTree(coords_domain)
        
        neigh = n
        dist, idx = tree.query( coords_i, 
                                k=neigh,
                                eps=0,
                                p=1,
                                distance_upper_bound=10)
    
        idx = idx.flatten()
        for i in idx:
          try:
            self.array[coords_domain[i]]=np.nan
          except IndexError:
            pass

    def edit(self):
      
        ny,nx = self.array.shape
        x_g, y_g = np.meshgrid(range(nx),range(ny))
        x_f, y_f = x_g.flatten(), y_g.flatten()
        coords_domain = zip(x_f, y_f)
        
        ''' find coords using kdTree (nearest neighbor)'''
        self.remove_line(coords_domain,az=59,n=3)
        self.remove_line(coords_domain,az=240,n=5)
        self.remove_ring(coords_domain,rang=116,n=30)

 
    def plot(self):

        ny, nx = self.array.shape
        
        x = np.arange(nx)
        y = np.arange(ny)
        
        arraym = ma.masked_where(np.isnan(self.array),self.array)
        array2m = ma.masked_where(np.isnan(self.array_in),self.array_in)

        fig,ax = plt.subplots(2,1,figsize=(5*1.5,8*1.5))
        

        ax[0].pcolormesh(x,y,array2m)
        ax[0].scatter(*self.origin,color='k')       
        ax[0].set_xlim([0,206])
        ax[0].set_ylim([0,182])
        
        ax[1].pcolormesh(x,y,arraym)
        ax[1].set_xlim([0,206])
        ax[1].set_ylim([0,182])
        ax[1].scatter(*self.origin,color='k')
        
        plt.show()


    def example(self):

        try:
          x09
        except NameError:
          x09=xta.process(case=[9])

        self.array = x09.ppi_ntta_z
        
        x = range(207)
        y = range(183)
        self.origin = (116,118)
        arraym = ma.masked_where(np.isnan(self.array),self.array)
        
        self.edit()
        
        array2m = ma.masked_where(np.isnan(self.array),self.array)
        
        
        fig,ax = plt.subplots(2,1,figsize=(5*1.5,8*1.5))
        
        ax[0].pcolormesh(x,y,arraym)
        ax[0].scatter(*self.origin,color='k')
        ax[0].scatter(116,50,color='r')
        ax[0].scatter(50,116,color='b')
        
        ax[0].set_xlim([0,206])
        ax[0].set_ylim([0,182])
        
        ax[1].pcolormesh(x,y,array2m)
        
        ax[1].set_xlim([0,206])
        ax[1].set_ylim([0,182])
        
        plt.show()        



def polar2cart(r, theta):
  
    x = r * np.cos(theta)
    y = r * np.sin(theta)*-1
    return x, y











