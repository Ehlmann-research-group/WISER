import sys
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src\\wiser")
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src")
from wiser.raster.dataset import RasterDataSet
from wiser.raster.loader import RasterDataLoader
from wiser.raster.roi import RegionOfInterest
from wiser.raster.selection import Selection, RectangleSelection, SinglePixelSelection, PolygonSelection, MultiPixelSelection
from wiser.raster.spectrum import calc_rect_spectrum, calc_roi_spectrum, calc_spectrum, calc_spectrum_fast
from PySide2.QtCore import *
import cProfile
import pstats
import time
import numpy as np

# Load in a raster dataset, a list of QPoints and spectrum.py
def calc_point(dataset_path: str):
    loader = RasterDataLoader()
    dataset = loader.load_from_file(dataset_path)
    roi = RegionOfInterest(name='testing_roi')
    # roi.add_selection(RectangleSelection(QPoint(10,800), QPoint(60, 850)))
    roi.add_selection(SinglePixelSelection(QPoint(10,800)))

    spectrum = calc_roi_spectrum(dataset, roi)


# Load in a raster dataset, a list of QPoints and spectrum.py
def calc_poly(dataset_path: str):
    loader = RasterDataLoader()
    dataset = loader.load_from_file(dataset_path)
    roi = RegionOfInterest(name='testing_roi')
    # roi.add_selection(RectangleSelection(QPoint(10,800), QPoint(60, 850)))
    list_obj = [QPoint(100,300), QPoint(100,290), QPoint(110, 320)]
    poly = PolygonSelection(list_obj)
    print("poly_length: ", len(poly.get_all_pixels()))
    roi.add_selection(poly)

    spectrum = calc_roi_spectrum(dataset, roi)


# Load in a raster dataset, a list of QPoints and spectrum.py
def calc_rect(dataset_path: str):
    loader = RasterDataLoader()
    dataset = loader.load_from_file(dataset_path)
    roi = RegionOfInterest(name='testing_roi')
    # roi.add_selection(RectangleSelection(QPoint(10,800), QPoint(310, 1100)))
    roi.add_selection(RectangleSelection(QPoint(75, 75), QPoint(125, 100)))

    # spectrum_fast, spectra_fast = calc_spectrum_fast(dataset, roi)
    spectrum_fast = calc_spectrum_fast(dataset, roi)
    print(f"fast) min: {np.nanmin(spectrum_fast)}, max: {np.nanmin(spectrum_fast)}, avg: {np.nanmean(spectrum_fast)}")
    # spectrum_normal, spectra_normal = calc_spectrum(dataset, roi.get_all_pixels())
    spectrum_normal = calc_spectrum(dataset, roi.get_all_pixels())
    print(f"normal) min: {np.nanmin(spectrum_normal)}, max: {np.nanmin(spectrum_normal)}, avg: {np.nanmean(spectrum_normal)}")
    assert(spectrum_fast.shape == spectrum_normal.shape)
    # print(f"Equal? {np.array_equal(spectrum_fast, spectrum_normal)}")
    # equal_spectra = np.array([np.allclose(row1, row2) for row1, row2 in zip(spectra_fast, spectra_normal)])
    # print(f"Equal element wise? {equal_spectra}")
    # print(f"Equal? {np.all(equal_spectra)}")
    # print(spectra_fast[0])
    # print(spectra_normal[0])
    # print(spectra_fast[0]-spectra_normal[0])


# Load in a raster dataset, a list of QPoints and spectrum.py
def profile(dataset_path: str):
    loader = RasterDataLoader()
    dataset = loader.load_from_file(dataset_path)
    roi = RegionOfInterest(name='testing_roi')
    roi.add_selection(RectangleSelection(QPoint(10,800), QPoint(310, 1100)))

    profiler = cProfile.Profile()
    profiler.enable()
    print('================Enabled Profile================')
    spectrum = calc_roi_spectrum(dataset, roi)
    profiler.disable()
    print('================Disabled Profile================')

    # Save the profiling stats to a file
    with open(f"output/roi_profile.txt", "w") as f:
        ps = pstats.Stats(profiler, stream=f)
        ps.sort_stats("tottime")
        ps.print_stats()

def profile_rect(dataset_path: str):
    loader = RasterDataLoader()
    dataset = loader.load_from_file(dataset_path)
    rect = QRect(QPoint(10,800), QSize(300, 300))

    profiler = cProfile.Profile()
    profiler.enable()
    print('================Enabled Profile================')
    spectrum = calc_rect_spectrum(dataset, rect)
    profiler.disable()
    print('================Disabled Profile================')

    # Save the profiling stats to a file
    with open(f"output/rect_profile.txt", "w") as f:
        ps = pstats.Stats(profiler, stream=f)
        ps.sort_stats("tottime")
        ps.print_stats()

def speed_test(dataset_path):
    loader = RasterDataLoader()
    dataset = loader.load_from_file(dataset_path)
    roi = RegionOfInterest(name='testing_roi')
    # roi.add_selection(RectangleSelection(QPoint(10,800), QPoint(60, 850)))
    list_obj = [QPoint(100,300), QPoint(300,500), QPoint(500, 300)]
    poly = PolygonSelection(list_obj)
    print("poly_length: ", len(poly.get_all_pixels()))
    roi.add_selection(poly)

    start_time_fast = time.time()
    spectrum = calc_roi_spectrum(dataset, roi)
    end_time_fast = time.time()

    start_time_previous_method = time.time()
    spectrum = calc_spectrum(dataset, roi.get_all_pixels())
    end_time_previous_method = time.time()

    print(f"Took {end_time_fast-start_time_fast} for new method")
    print(f"Took {end_time_previous_method-start_time_previous_method} for previous method")


if __name__ == '__main__':
    dataset_path = 'C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\Task1.2_Slow_ROI_Mean_5gb_285_spectra\\RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr_expanded_lines_and_samples_2.hdr'
    dataset_path = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\ang20171108t184227_corr_v2p13_subset_bil.hdr"
    # profile(dataset_path)
    # profile_rect(dataset_path)
    # calc_poly(dataset_path)
    calc_rect(dataset_path)

    # speed_test(dataset_path)
    print('Done with profiling')

'''
Commands:
- py-spy, sample profiler
    - From src/wiser/profiling
    - py-spy record -o output/profile.svg -- python -m spectra_mean.py
- cProfile
    - From src/wiser/profiling
    - python -m cProfile spectra_mean.py > output/profile_output.txt
'''
