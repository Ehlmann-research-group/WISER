import sys
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src\\wiser")
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src")
from wiser.raster.dataset import RasterDataSet
from wiser.raster.loader import RasterDataLoader
from wiser.raster.roi import RegionOfInterest
from wiser.raster.selection import RectangleSelection
from wiser.raster.spectrum import calc_roi_spectrum
# from PySide2.QtCore import *
from wiser.bandmath.types import BandMathValue, VariableType
from wiser.bandmath.builtins import OperatorAdd, OperatorCompare, OperatorDivide, OperatorMultiply, OperatorPower, OperatorSubtract, OperatorUnaryNegate
from wiser.gui.app import DataVisualizerApp

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from wiser.gui.rasterview import make_channel_image
from wiser.raster.stretch import StretchLog2, StretchHistEqualize, StretchComposite, StretchLinear, StretchSquareRoot

import cProfile
import pstats

def profile(dataset_path: str):
    app = QApplication([])

    #========================================================================
    # WISER Application Initialization


    # Set the initial window size to be 70% of the screen size.
    wiser_ui = DataVisualizerApp()
    screen_size = app.screens()[0].size()
    wiser_ui.resize(screen_size * 0.7)
    wiser_ui.show()
    app_state = wiser_ui._app_state
    loader = RasterDataLoader()
    dataset1 = loader.load_from_file(dataset_path)

    app_state.add_dataset(dataset1)
    # profiler = cProfile.Profile()
    # profiler.enable()
    # print('================Enabled Profile================')
    wiser_ui._main_view.get_stretch_builder()._stretch_config._ui.rb_cond_none.setChecked(True)
    wiser_ui._main_view.get_stretch_builder()._stretch_config._ui.rb_cond_sqrt.setChecked(True)
    wiser_ui._main_view.get_stretch_builder()._on_conditioner_type_changed()

    # profiler.disable()
    # print('================Disabled Profile================')
    # # Save the profiling stats to a file
    # with open("output/bandmath_add_stats.txt", "w") as f:
    #     ps = pstats.Stats(profiler, stream=f)
    #     ps.sort_stats("tottime") 
    #     ps.print_stats()
    # sys.exit(app.exec_())

def profile2(dataset_path: str):
    loader = RasterDataLoader()
    dataset1 = loader.load_from_file(dataset_path)
    log_stretch = StretchLog2()

    profiler = cProfile.Profile()
    profiler.enable()
    print('================Enabled Profile================')
    make_channel_image(dataset1, 1, log_stretch)
    profiler.disable()
    print('================Disabled Profile================')
    # Save the profiling stats to a file
    with open("output/hist_log_stats.txt", "w") as f:
        ps = pstats.Stats(profiler, stream=f)
        ps.sort_stats("tottime") 
        ps.print_stats()


if __name__ == '__main__':
    '''
    It is okay for this profile to use an image that is not incredibly big because the sampler profile takes a long time to run
    '''
    # dataset_path = "c:\\Users\\jgarc\\OneDrive\\Documents\\Data\\ang20171108t184227_corr_v2p13_subset_bil.hdr"
    # dataset_path = 'C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr'
    large_bands = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\C5705B-00003Z-01_2018_07_28_14_18_38_VNIRcalib.hdr"
    # dataset_path = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\Task1.1_SlowBandMath_10gb\\ang20171108t184227_corr_v2p13_subset_bil_expanded_bands_by_40.hdr"
    # dataset_path = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\Task1.1_SlowBandMath_10gb\\ang20171108t184227_corr_v2p13_subset_bil_increased_bands_by_80.hdr"

    profile2(large_bands)
    # profile_cube_band(dataset_path)
    # profile_cube_spectrum(dataset_path)
    print('Done with profiling')
