import sys
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src\\wiser")
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src")
from wiser.raster.loader import RasterDataLoader
from wiser.gui.app import DataVisualizerApp

from wiser.gui.stretch_builder import StretchBuilderDialog

from PySide2.QtWidgets import QApplication
import cProfile
import pstats

# Load in a raster dataset, a list of QPoints and spectrum.py
def profile(dataset_path: str, output_path: str) -> None:
    loader = RasterDataLoader()
    dataset = loader.load_from_file(dataset_path)

    # stretch_builder_dialog = StretchBuilderDialog()
    app = QApplication([])
    app = DataVisualizerApp()
    app._app_state.open_file(dataset_path)
    app._main_view._on_stretch_builder()
    profiler = cProfile.Profile()
    profiler.enable()
    print('================Enabled Profile================')
    app._main_view._stretch_builder._stretch_config._ui.rb_stretch_linear.click()
    profiler.disable()
    print('================Disabled Profile================')

    # TODO(donnie):  Pass Qt arguments
    # Save the profiling stats to a file
    with open(output_path, "w") as f:
        ps = pstats.Stats(profiler, stream=f)
        ps.sort_stats("tottime")
        ps.print_stats()

if __name__ == '__main__':
    dataset_path = 'C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr'
    output_path = 'C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src\\wiser\\profiling\\output\\constrast_stretch_stats.txt'
    profile(dataset_path, output_path)
    print('Done with profiling')