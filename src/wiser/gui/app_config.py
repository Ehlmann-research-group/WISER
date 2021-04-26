import enum
import json
import os
import platform

from typing import Any, Dict, List, Optional

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from wiser import version
from wiser.raster.spectrum import SpectrumAverageMode



class PixelReticleType(enum.Enum):
    '''
    This enumeration specifies the different options for how a selected pixel
    is highlighted in the user interface.
    '''

    # Draw a "small cross" - the horizontal and vertical lines will only have
    # a relatively small extent.
    SMALL_CROSS = 1

    # Draw a "large cross" - the horizontal and vertical lines will extend to
    # the edges of the view.
    LARGE_CROSS = 2

    # Draw a "small cross" at low magnifications, but above a certain
    # magnification level (e.g. 4x), start drawing a box around the selected
    # pixel.
    SMALL_CROSS_BOX = 3


class LegendPlacement(enum.Enum):
    '''
    An enumeration of the placement options that the spectral plot window
    recognizes.  These are mapped to matplotlib arguments in the component.
    '''
    NO_LEGEND = 0

    UPPER_LEFT = 1
    UPPER_CENTER = 2
    UPPER_RIGHT = 3
    CENTER_LEFT = 4
    CENTER_RIGHT = 5
    LOWER_LEFT = 6
    LOWER_CENTER = 7
    LOWER_RIGHT = 8

    OUTSIDE_CENTER_RIGHT = 20
    OUTSIDE_LOWER_CENTER = 30

    BEST_LOCATION = 50


def get_wiser_config_dir() -> str:
    '''
    Determine the WISER config directory based on the platform/OS.  The config
    directory is as follows:

    *   On Windows, the user's `AppData\Local\WISER` directory is used.
    *   On macOS, the user's `~/Library/WISER` directory is used.
    *   All other platforms are assumed to be Linux/*NIX, and `~/.wiser` is
        used.  Note that a warning is output if the platform is not Linux.
    '''
    sys_name = platform.system()
    if sys_name == 'Windows':
        # Put WISER config in the user's local application data directory
        wiser_dir = os.path.expandvars(r'%LOCALAPPDATA%\WISER')

    elif sys_name == 'Darwin':
        # Put WISER config in the user's Library directory
        wiser_dir = os.path.expanduser('~/Library/WISER')

    else:
        # Use the
        if sys_name != 'Linux':
            warnings.warn(f'Unrecognized platform name "{sys_name}"')

        wiser_dir = os.path.expanduser('~/.wiser')

    return wiser_dir

"""
def get_path_to_wiser_conf(*subpaths: List[str]):
    '''
    This function returns the full path to a WISER configuration file (or
    subdirectory) given the subpath of the config-file within the WISER
    configuration directory.  The base config directory path depends on the
    operating system, since Windows, macOS and Linux all have different
    conventions for where configuration is stored.

    The `subpaths` argument may be one or more subpaths that are all
    concatenated together after the base WISER config path.  The Python
    `os.path.join()` function is used to do the path joining, so any argument
    that function can take, may be passed to this function.
    '''
    return os.path.join(get_wiser_config_dir(), subpaths)
"""

class ApplicationConfig:

    DEFAULTS = {
        # General properties - these are all scalars

        'general.version'              : (str, version.VERSION),
        'general.online_bug_reporting' : (bool, False),
        'general.red_wavelength_nm'    : (int, 700),
        'general.green_wavelength_nm'  : (int, 530),
        'general.blue_wavelength_nm'   : (int, 470),

        'raster.pixel_cursor_type'        : (str, 'SMALL_CROSS'), # PixelReticleType
        'raster.pixel_cursor_color'       : (str, 'red'),
        'raster.viewport_highlight_color' : (str, 'yellow'),

        'raster.selection.edit_outline'   : (str, 'white'),
        'raster.selection.edit_points'    : (str, 'yellow'),

        'spectra.default_area_avg_x'    : (int, 1),
        'spectra.default_area_avg_y'    : (int, 1),
        'spectra.default_area_avg_mode' : (str, 'MEAN'), # SpectrumAverageMode

        # Plugin configuration - by default this is all empty

        'plugin_paths' : (list, []),
        'plugins' : (list, []),
    }

    FEATURE_FLAGS = 'feature_flags.'

    def __init__(self):
        '''
        Initialize a new ApplicationConfig object with default settings.
        '''
        self._config: Dict = {}
        self._load_defaults()


    def _load_defaults(self):
        '''
        This helper function generates and returns a default WISER configuration.
        Any user-specific configuration is loaded on top of this, so that all
        key options are always specified, even if user configuration is
        incomplete.
        '''

        self._config = {}
        for (name, (value_type, default_value)) in self.DEFAULTS.items():
            self.set(name, default_value)


    def get(self, option, default=None, as_type=None) -> Any:
        '''
        Returns the value of the specified config option.  An optional default
        value may be specified.

        Options are specified as a sequence of names separated by dots '.',
        just like a series of object-member accesses on an object hierarchy.
        '''
        option = option.strip().lower()

        value = self._config.get(option)
        if value is None:
            return default

        if as_type is None:
            # Choose a default type for the option's value, either based on the
            # defaults, or if it is a feature flag then choose bool.

            if option in self.DEFAULTS:
                as_type = self.DEFAULTS[option][0]

            elif option.startswith(self.FEATURE_FLAGS):
                as_type = bool

        if as_type is not None:
            value = as_type(value)

        return value


    def set(self, option, value):
        '''
        Sets the value of the specified config option.
        '''
        option = option.strip().lower()

        # If we know about this option, or if it is a feature-flag, convert it
        # to the expected value-type.
        if option in self.DEFAULTS:
            as_type = self.DEFAULTS[option][0]
            value = as_type(value)
        elif option.startswith(self.FEATURE_FLAGS):
            value = bool(value)

        self._config[option] = value


    def load(self, config_path: str) -> None:
        '''
        Load configuration from the specified file path into this object.  If
        an exception is raised, it will propagate out of this function, and the
        current contents of this object will remain unchanged.  If the load
        completes successfully, this object will contain the configuaration from
        the data file, with any missing values provided by the default
        configuration settings.
        '''

        # Load the file!  If an exception occurs, let it propagate out.
        with open(config_path) as f:
            file_conf = json.load(f)

        # Make sure the loaded file configuration is an object of key-value
        # pairs.

        if not isinstance(file_conf, dict):
            raise ValueError('Loaded configuration must be a Python dict')

        for key, value in file_conf.items():
            if not isinstance(key, str):
                raise ValueError('Loaded configuration must have string keys')

            # TODO(donnie):  The plugin config is lists.
            # if isinstance(value, dict) or isinstance(value, list):
            #     raise ValueError('Loaded configuration must have scalar values')

        # If we got here, we were able to successfully load the configuration
        self._load_defaults()
        for key, value in file_conf.items():
            self.set(key, value)


    def save(self, config_path: str) -> None:
        '''
        Save the configuration in this object to the specified file path.  If
        an exception is raised, it will propagate out of this function.
        '''
        # Make sure the target directory exists first.
        folder_path = os.path.dirname(config_path)
        if folder_path:
            os.makedirs(folder_path, exist_ok=True)

        # Save the file.  Let exceptions propagate out.
        with open(config_path, 'w') as f:
            json.dump(self._config, f, sort_keys=True, indent=4)


    def to_string(self) -> str:
        return json.dumps(self._config, sort_keys=True, indent=4)
