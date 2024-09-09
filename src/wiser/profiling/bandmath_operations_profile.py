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

    profiler = cProfile.Profile()
    profiler.enable()
    print('================Enabled Profile================')
    res = OperatorAdd().apply([lhs, rhs])
    profiler.disable()
    print('================Disabled Profile================')
    # Save the profiling stats to a file
    with open("output/bandmath_add_stats.txt", "w") as f:
        ps = pstats.Stats(profiler, stream=f)
        ps.sort_stats("tottime") 
        ps.print_stats()


if __name__ == '__main__':
    dataset_path = 'C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr'
    profile(dataset_path)
    print('Done with profiling')
