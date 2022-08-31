# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: bav

Collection of GIS-related functions

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import rasterio
from rasterio import features
import geopandas as gpd
import earthpy.plot as ep
from rasterio.plot import plotting_extent
import rasterstats as rs
import rasterio as rio
import numpy as np
import pandas as pd
import os.path as path


def shape_to_binary_raster(shape_in, raster_in):
    # shapefile/geopandas to binary raster
    # converts a shapefile (given as path to shapefile or geopandas) into a binary
    # raster given a template raster (given as path to tif or rasterio reader)
    # Shapefile and template raster should have the same CRS.

    if isinstance(shape_in, str):
        shape_in = gpd.GeoDataFrame.from_file(shape_in)
        # shape_in = ice.to_crs('EPSG:3413')

    if isinstance(raster_in, str):
        raster_in = rasterio.open(raster_in)

    meta = raster_in.meta.copy()
    meta.update(compress="lzw")

    out_arr = raster_in.read(1)

    # this is where we create a generator of geom, value pairs to use in rasterizing
    shapes = ((geom, 1) for geom in shape_in.geometry)
    burned = features.rasterize(
        shapes=shapes, fill=0, out=out_arr, transform=raster_in.transform
    )
    return burned == 1


def write_geotiff(var, name_out, ref_raster_info, deflate=True):
    # write raster based on reference raster
    if isinstance(ref_raster_info, str):
        ref_file = rasterio.open(ref_raster_info)
        # ref_file = rio.open('MEASURES melt/grid_domain_3413.tif')
        ref_raster_info = ref_file.meta
    if deflate:
        with rasterio.Env():
            ref_raster_info.update(compress="DEFLATE")
    ref_raster_info.update(dtype=var.dtype)
    with rasterio.open(name_out, "w+", **ref_raster_info) as dst:
        dst.write(var, 1)


from rasterio.warp import transform
import rioxarray


def get_lat_lon_from_raster(filename, write_files=True):
    da = rioxarray.open_rasterio(filename).squeeze()
    da = da.rio.write_crs(3413)

    # Compute the lon/lat coordinates with rasterio.warp.transform
    ny, nx = len(da["y"]), len(da["x"])
    x, y = np.meshgrid(da["x"], da["y"])

    # Rasterio works with 1D arrays
    lon, lat = transform(da.rio.crs, {"init": "EPSG:4326"}, x.flatten(), y.flatten())
    lon = np.asarray(lon).reshape((ny, nx))
    lat = np.asarray(lat).reshape((ny, nx))

    if write_files:
        da.values = lon
        da.rio.to_raster("lon.tif")
        da.values = lat
        da.rio.to_raster("lat.tif")
    return lat, lon


def plot_geotiff(raster_in, ax, cmap, vmin, vmax, cbar_label):
    raster = raster_in.read(1)
    raster[raster == raster_in.nodata] = np.nan
    tmp2, cbar2 = ep.plot_bands(
        raster,
        extent=plotting_extent(raster_in),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )
    cbar2.ax.set_ylabel(cbar_label, fontsize=16)
    cbar2.ax.tick_params(axis="y", labelsize=16)
    return tmp2, cbar2


def zonal_stat(
    geotiff_file_name,
    min_val=None,
    max_val=None,
    buffer_shape_file="Data/AWS/AWS_buffer.shp",
):
    # extract raster statistics in given polygons from a geopanda
    with rio.open(geotiff_file_name) as src:
        data_raster = src.read(1)
        data_raster_meta = src.profile

    if min_val:
        data_raster[data_raster < min_val] = np.nan
    if max_val:
        data_raster[data_raster > max_val] = np.nan
    # Extract zonal stats
    zonal_stat_out = rs.zonal_stats(
        buffer_shape_file,
        data_raster,
        affine=data_raster_meta["transform"],
        geojson_out=True,
        raster_out=True,
        copy_properties=True,
        stats="count std median",
    )
    return zonal_stat_out


def sample_raster_with_geopandas(shape_file_name, raster_file_name, field_name):
    # Read points from shapefile
    if isinstance(shape_file_name, str) and path.exists(shape_file_name):
        pts = gpd.read_file(shape_file_name)
        pts.columns = ["name", "lat", "lon", "elev", "geometry"]
    elif isinstance(shape_file_name, pd.DataFrame):
        pts = shape_file_name
    else:
        print("Cannot find " + shape_file_name)
        return []
    pts.index = range(len(pts))
    coords = [(x, y) for x, y in zip(pts["geometry"].x, pts["geometry"].y)]

    # Open the raster and store metadata
    if path.exists(raster_file_name):
        src = rasterio.open(raster_file_name)

        # Sample the raster at every point location and store values in DataFrame
        pts[field_name] = [x for x in src.sample(coords)]
        pts[field_name] = pts.apply(lambda x: x[field_name][0], axis=1)
        return pts[field_name].values
    else:
        print("Cannot find " + raster_file_name)

    return []


from rasterio.features import shapes


def polygonize_raster(filename, mask=None):
    with rio.Env():
        with rio.open(filename) as src:
            image = src.read(1).astype("int16")  # first band
            results = (
                {"properties": {"raster_val": v}, "geometry": s}
                for i, (s, v) in enumerate(
                    shapes(image, mask=mask, transform=src.transform)
                )
            )
    geoms = list(results)
    gpd_polygonized_raster = gpd.GeoDataFrame.from_features(geoms)
    geoms = gpd_polygonized_raster.geometry.values  # list of shapely geometries
    geometry = geoms[0]  # shapely geometry
    # transform to GeJSON format
    geoms = [mapping(geoms[0])]
    return gpd_polygonized_raster


from rasterio.mask import mask
from shapely.geometry import mapping
from rasterio import Affine  # or from affine import Affine
from shapely.geometry import Point
import matplotlib.pyplot as plt


def clip_raster_with_shapefile(raster_file, shapefile, only_values=True):
    if isinstance(shapefile, str):
        shapefile = gpd.read_file(shapefile)
    # extract the geometry in GeoJSON format
    all_geoms = shapefile.geometry.values  # list of shapely geometries

    # transform to GeJSON format
    for i, geoms in enumerate(all_geoms):
        geoms = [mapping(geoms)]
        # x,y = polygon1.exterior.xy
        # plt.plot(x,y)
        # extract the raster values values within the polygon
        with rio.open(raster_file) as src:
            out_image, out_transform = mask(src, geoms, crop=False)

        # extract the values of the masked array
        data = out_image[0]
        if np.all(np.isnan(data)):
            continue
        if only_values:
            # extract the row, columns of the valid values
            row, col = np.where(~np.isnan(data))
            val = np.extract(~np.isnan(data), data)
            d = pd.DataFrame({"col": col, "row": row, "val": val})

            if "d_all" not in locals():
                d_all = d
            else:
                d_all = d_all.append(d)
        else:
            # extract the row, columns of the valid values
            row, col = np.where(~np.isnan(data))
            val = np.extract(~np.isnan(data), data)

            # Now I use How to I get the coordinates of a cell in a geotif? or Python affine transforms to transform between the pixel and projected coordinates with out_transform as the affine transform for the subset data
            T1 = out_transform * Affine.translation(
                0.5, 0.5
            )  # reference the pixel centre
            rc2xy = lambda r, c: (c, r) * T1

            # Creation of a new resulting GeoDataFrame with the col, row and elevation values
            d = gpd.GeoDataFrame({"col": col, "row": row, "val": val})

            # coordinate transformation
            d["x"] = d.apply(lambda row: rc2xy(row.row, row.col)[0], axis=1)
            d["y"] = d.apply(lambda row: rc2xy(row.row, row.col)[1], axis=1)
            # geometry
            d["geometry"] = d.apply(lambda row: Point(row["x"], row["y"]), axis=1)
            if "d_all" not in locals():
                d_all = d
            else:
                d_all = d_all.append(d)
    clipped_raster = rio.open(raster_file).read(1) * np.nan
    anti_clipped_raster = rio.open(raster_file).read(1)
    for k, l, val in zip(d_all.row, d_all.col, d_all.val):
        clipped_raster[k, l] = val
        anti_clipped_raster[k, l] = np.nan
    return d_all, clipped_raster, anti_clipped_raster
