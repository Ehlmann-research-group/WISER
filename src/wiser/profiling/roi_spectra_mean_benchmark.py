import sys
import os
from typing import List
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src\\wiser")
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src")
from wiser.raster.dataset import RasterDataSet
from wiser.raster.loader import RasterDataLoader
from wiser.raster.roi import RegionOfInterest
from wiser.raster.selection import Selection, RectangleSelection, SinglePixelSelection, PolygonSelection, MultiPixelSelection
from wiser.raster.spectrum import calc_rect_spectrum, calc_roi_spectrum, calc_spectrum_testing, calc_spectrum_fast
from wiser.raster import roi_export
from PySide2.QtCore import *
import cProfile
import pstats
import time
import numpy as np

def get_hdr_files(folder_path):
    hdr_files = []
    # Walk through the directory
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.hdr'):
                # Get absolute file path and append to the list
                hdr_files.append(os.path.join(os.path.abspath(root), file))
    return hdr_files

def calc_func_speed(hdr_paths: List[str], func, roi: RegionOfInterest):
    '''calculates the speed that it takes for an average roi to be calculated for all the hdrs passed in'''
    loader = RasterDataLoader()
    times = []
    for hdr_path in hdr_paths:
        print(f"Running the file: {hdr_path}")
        dataset = loader.load_from_file(hdr_path)
        start_time = time.time()
        func(dataset, roi)
        end_time = time.time()
        times.append(end_time-start_time)
    return times

def make_grid_list(start, stop, step):
    # Got multi pixel points that are connected
    x = np.arange(start, stop, step) 
    y = np.arange(start, stop, step)

    X, Y = np.meshgrid(x, y)

    points = np.column_stack([X.ravel(), Y.ravel()])
    return [tuple(point) for point in points.tolist()]

def get_roi_speed_stats(hdr_paths):
    rect_roi = RegionOfInterest()
    rect_roi.add_selection(RectangleSelection(QPoint(10, 10), QPoint(60, 60)))
    poly_points = [QPoint(10,10), QPoint(60,10), QPoint(40, 40)]
    poly_roi = RegionOfInterest()
    poly_roi.add_selection(PolygonSelection(poly_points))


    # Convert to a Python list if needed
    points_connected_list = make_grid_list(10, 21, 1)
    points_connected_list = [QPoint(point[0], point[1]) for point in points_connected_list]
    points_disjoint_list = make_grid_list(10, 51, 2)
    points_disjoint_list = [QPoint(point[0], point[1]) for point in points_disjoint_list]
    print(f"len points_connected_list: {len(points_connected_list)}")
    print(f"len points_disjoint_list: {len(points_disjoint_list)}")
    multi_pix_points_connected = RegionOfInterest()
    multi_pix_points_connected.add_selection(MultiPixelSelection(points_connected_list))
    multi_pix_points_disjoint = RegionOfInterest()
    multi_pix_points_disjoint.add_selection(MultiPixelSelection(points_disjoint_list))
    
    rois = {}
    rois['rect_roi'] = rect_roi
    rois['poly_roi'] = poly_roi
    rois['multi_pix_points_connected'] = multi_pix_points_connected
    rois['multi_pix_points_disjoint'] = multi_pix_points_disjoint
    results = {}
    for roi_name, roi in rois.items():
        print(f"================Starting fast_times for {roi_name}================")
        fast_times = calc_func_speed(hdr_paths, calc_spectrum_fast, roi)
        print(f"================Slow Times fast_times for {roi_name}================")
        slow_times = calc_func_speed(hdr_paths, calc_spectrum_testing, roi)
        results[roi_name] = (fast_times, slow_times)

    with open('output/benchmark_roi_avg.txt', 'w') as f:  # Open file in write mode
        for roi_name, (fast_times, slow_times) in results.items():
            line = f"For {roi_name}\n\tNew method avg: {np.mean(fast_times)}\n\tOld method avg: {np.mean(slow_times)}"
            print(line)
            f.write(line + '\n\n')  # Write to file with a newline character

if __name__ == '__main__':
    folder_path = 'C:\\Users\jgarc\\OneDrive\\Documents\\Data\\Benchmarks'
    hdr_files = get_hdr_files(folder_path)
    print("hdr_files: ", hdr_files)
    get_roi_speed_stats(hdr_files)
    