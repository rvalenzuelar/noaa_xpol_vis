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
  
    def __init__(self,array=None,qc_list=None):
      
      self.array_in = array.copy()
      self.array = array.copy()

      if array is not None and qc_list is not None:
          self.edit(qc_list)

    def get_edited(self):
        
        return self.array


    def remove_line(self,coords_domain,az=None,n=None):
        
        maxrange = 200
        direction = az * (np.pi/180.)
        
        r = np.arange(maxrange)
        theta = np.array([direction]*maxrange)
        
        setNans_kdTree(self,r,theta,n,coords_domain)        
        
    def remove_ring(self,coords_domain,rang=None,n=None):
    
        
        theta = np.linspace(0,2*np.pi,300)
        r = np.array([rang]*theta.size)
        
        setNans_kdTree(self,r,theta,n,coords_domain)


    def edit(self,qc_list):
      
        ny,nx = self.array.shape
        x_g, y_g = np.meshgrid(range(ny),range(nx))
        x_f, y_f = x_g.flatten(), y_g.flatten()
        coords_domain = zip(x_f, y_f)
        
        for qc in qc_list:
            self.origin = qc['origin']
            n = qc['n']
            if qc['target'] == 'remove_ring':
                self.remove_ring(coords_domain,
                                 rang=qc['rang'],
                                 n=n)        
            elif qc['target'] == 'remove_line':
                self.remove_line(coords_domain,
                                 az=qc['az'],
                                 n=n)

 
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


def setNans_kdTree(self,r,theta,n,coords_domain):
    
        xi,yi=polar2cart(r, theta)
        xi+=self.origin[0]
        yi+=self.origin[1]
        coords_i = zip(xi,yi)  

        ''' create kdTree with entire domain'''
        tree = cKDTree(coords_domain)
        
        ''' determine indices of polar coordinates '''
        neigh = n
        dist, idx = tree.query( coords_i, 
                                k=neigh,
                                eps=0,
                                p=1,
                                distance_upper_bound=10)
    
        ''' set values of indices to NaN'''
        idx = idx.flatten()
        for i in idx:
          try:
            self.array[coords_domain[i]]=np.nan
          except IndexError:
            pass
    








