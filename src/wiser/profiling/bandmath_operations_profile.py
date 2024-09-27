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
import cProfile
import pstats

def profile(dataset_path: str):
    
    loader = RasterDataLoader()
    dataset1 = loader.load_from_file(dataset_path)
    dataset2 = loader.load_from_file(dataset_path)

    lhs = BandMathValue(VariableType.IMAGE_CUBE, dataset1)
    rhs = BandMathValue(VariableType.IMAGE_CUBE, dataset2)

    # profiler = cProfile.Profile()
    # profiler.enable()
    # print('================Enabled Profile================')
    res = OperatorAdd().apply([lhs, rhs], 0)
    # profiler.disable()
    # print('================Disabled Profile================')
    # # Save the profiling stats to a file
    # with open("output/bandmath_add_stats.txt", "w") as f:
    #     ps = pstats.Stats(profiler, stream=f)
    #     ps.sort_stats("tottime") 
    #     ps.print_stats()

def profile_cube_band(dataset_path: str):
    
    loader = RasterDataLoader()
    dataset1 = loader.load_from_file(dataset_path)
    dataset2 = loader.load_from_file(dataset_path)

    lhs = BandMathValue(VariableType.IMAGE_CUBE, dataset1)
    rhs = BandMathValue(VariableType.IMAGE_BAND, dataset2.get_band_data(0))

    # profiler = cProfile.Profile()
    # profiler.enable()
    # print('================Enabled Profile================')
    res = OperatorAdd().apply([lhs, rhs])
    # profiler.disable()
    # print('================Disabled Profile================')
    # # Save the profiling stats to a file
    # with open("output/bandmath_add_stats.txt", "w") as f:
    #     ps = pstats.Stats(profiler, stream=f)
    #     ps.sort_stats("tottime") 
    #     ps.print_stats()

def profile_cube_spectrum(dataset_path: str):
    
    loader = RasterDataLoader()
    dataset1 = loader.load_from_file(dataset_path)
    dataset2 = loader.load_from_file(dataset_path)

    lhs = BandMathValue(VariableType.IMAGE_CUBE, dataset1)
    rhs = BandMathValue(VariableType.SPECTRUM, dataset2.get_all_bands_at(50, 50))

    # profiler = cProfile.Profile()
    # profiler.enable()
    # print('================Enabled Profile================')
    res = OperatorAdd().apply([lhs, rhs])
    # profiler.disable()
    # print('================Disabled Profile================')
    # # Save the profiling stats to a file
    # with open("output/bandmath_add_stats.txt", "w") as f:
    #     ps = pstats.Stats(profiler, stream=f)
    #     ps.sort_stats("tottime") 
    #     ps.print_stats()

if __name__ == '__main__':
    '''
    It is okay for this profile to use an image that is not incredibly big because the sampler profile takes a long time to run
    '''
    dataset_path = "c:\\Users\\jgarc\\OneDrive\\Documents\\Data\\ang20171108t184227_corr_v2p13_subset_bil.hdr"
    # dataset_path = 'C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr'
    # dataset_path = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\C5705B-00003Z-01_2018_07_28_14_18_38_VNIRcalib.hdr"
    # dataset_path = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\Task1.1_SlowBandMath_10gb\\ang20171108t184227_corr_v2p13_subset_bil_expanded_bands_by_40.hdr"
    # dataset_path = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\Task1.1_SlowBandMath_10gb\\ang20171108t184227_corr_v2p13_subset_bil_increased_bands_by_80.hdr"
    profile(dataset_path)
    # profile_cube_band(dataset_path)
    # profile_cube_spectrum(dataset_path)
    print('Done with profiling')

    '''
    Timings
    original takes 857 secds on the 15GB image
    old takes _______ on the 15GB image
    new takes _______ on the 15GB image
    '''

