import matplotlib.tri as tri
from matplotlib.tri import UniformTriRefiner
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

class Generate_Contour():
    def __init__(self, level, smooth_level):
        self.levels = level
        self.smooth_level = smooth_level

    def TriContour(self, x, y, v):
        # Generate the contour plot with specified levels
        triang = tri.Triangulation(x, y)
        refiner = UniformTriRefiner(triang)
        if self.smooth_level is not None:
            tri_refined, property_refined = refiner.refine_field(v, subdiv=self.smooth_level)
        else:
            tri_refined, property_refined = refiner.refine_field(v, subdiv=3)
        
        if self.levels is not None:
            if isinstance(self.levels, int):
                plt.tricontour(tri_refined, property_refined, levels=self.levels, colors="k", linewidths=0.5)
                plt.tricontourf(tri_refined, property_refined, levels=self.levels, cmap='viridis')
            else:
                plt.tricontour(tri_refined, property_refined, colors="k", linewidths=0.5)
                plt.tricontourf(tri_refined, property_refined, cmap='viridis')

        plt.colorbar()
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

    def CartContour(self, x, y, v):
        # Create a grid for Cartesian coordinates
        xi = np.linspace(min(x), max(x), 100)
        yi = np.linspace(min(y), max(y), 100)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate data onto the grid
        if self.smooth_level is not None:
            vi = griddata((x, y), v, (xi, yi), method='cubic')
        else:
            vi = griddata((x, y), v, (xi, yi), method='linear')

        # Generate the contour plot with specified levels
        if self.levels is not None:
            if isinstance(self.levels, int):
                plt.contour(xi, yi, vi, levels=self.levels, colors="k", linewidths=0.5)
                plt.contourf(xi, yi, vi, levels=self.levels, cmap='viridis')
            else:
                plt.contour(xi, yi, vi, colors="k", linewidths=0.5)
                plt.contourf(xi, yi, vi, cmap='viridis')

        plt.colorbar()
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')