import numpy as np

from wiser.plugins.types import BatchProcessingPlugin, BatchProcessingInputType, BatchProcessingOutputType

class AverageBandsPlugin(BatchProcessingPlugin):
    def __init__(self):
        super().__init__()

    def get_ordered_input_types(self):
        return [BatchProcessingInputType.IMAGE_BAND, BatchProcessingInputType.IMAGE_BAND]
    
    def get_ordered_output_types(self):
        return [BatchProcessingOutputType.IMAGE_BAND]
    
    def process(self, *args):
        band1 = args[0]
        band2 = args[1]

        return (band1 + band2) / 2
