from wiser.profiling.roi_spectra_mean_profile import test_calc_spectrum

if __name__ == '__main__':
    print('Running')
    dataset_path = 'C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr'
    test_calc_spectrum(dataset_path)
    print('Done with calculation')