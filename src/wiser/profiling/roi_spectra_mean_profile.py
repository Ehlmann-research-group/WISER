import sys
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src\\wiser")
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src")
from wiser.raster.dataset import RasterDataSet
from wiser.raster.loader import RasterDataLoader
from wiser.raster.roi import RegionOfInterest
from wiser.raster.selection import RectangleSelection
from wiser.raster.spectrum import calc_roi_spectrum
from PySide2.QtCore import *
import cProfile
import pstats

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

if __name__ == '__main__':
    dataset_path = 'C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\Task1.2_Slow_ROI_Mean_5gb_285_spectra\\RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr_expanded_lines_and_samples_2.hdr'
    profile(dataset_path)
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
