from typing import List, Tuple, Optional
import tifffile
import h5py
import pathlib
import numpy as np
from ophys_etl.utils.array_utils import normalize_array
from ophys_etl.modules.mesoscope_splitting.mixins import (
    IntFromZMapperMixin)
from ophys_etl.modules.mesoscope_splitting.tiff_metadata import (
    ScanImageMetadata)


class ScanImageTiffSplitter(IntFromZMapperMixin):
    """
    A class to naively split up a tiff file by just looping over
    the scanfields in its ROIs

    **this will not work for z-stacks**

    Parameters
    ----------
    tiff_path: pathlib.Path
        Path to the TIFF file whose metadata we are parsing
    """

    def __init__(self, tiff_path: pathlib.Path):
        self._file_path = tiff_path
        self._metadata = ScanImageMetadata(
                             tiff_path=tiff_path)

        self._validate_z_stack()
        self._get_z_manifest()
        self._frame_shape = dict()

    def _validate_z_stack(self):
        """
        Make sure that the zsAllActuators are arranged the
        way we expect, i.e.
        [[roi0_z0, roi0_z1, roi0_z2..., roi0_zM],
         [roi0_zM+1, roi0_zM+2, ...],
         [roi1_z0, roi1_z1, roi1_z2..., roi1_zM],
         [roi1_zM+1, roi1_zM+2, ...],
         ...
         [roiN_z0, roiN_z1, roiN_z2...]]

        or, in the case of one ROI

        [z0, z1, z2....]
        """
        z_value_array = self._metadata.all_zs()

        # check that z_value_array is a list of lists
        if not isinstance(z_value_array, list):
            msg = "Unclear how to split this TIFF\n"
            msg += f"{self._file_path.resolve().absolute()}\n"
            msg += f"metadata.all_zs {self._metadata.all_zs()}"
            raise RuntimeError(msg)

        if isinstance(z_value_array[0], list):
            z_value_array = np.concatenate(z_value_array)
        defined_rois = self._metadata.defined_rois

        z_int_per_roi = []

        msg = ""

        # check that the same Z value does not appear more than
        # once in the same ROI
        for i_roi, roi in enumerate(defined_rois):
            if isinstance(roi['zs'], list):
                roi_zs = roi['zs']
            else:
                roi_zs = [roi['zs'], ]
            roi_z_ints = [self._int_from_z(z_value=zz)
                          for zz in roi_zs]
            z_int_set = set(roi_z_ints)
            if len(z_int_set) != len(roi_zs):
                msg += f"roi {i_roi} has duplicate zs: {roi['zs']}\n"
            z_int_per_roi.append(z_int_set)

        # check that z values in z_array occurr in ROI order
        offset = 0
        n_roi = len(z_int_per_roi)
        n_z_per_roi = len(z_value_array)//n_roi

        # check that every ROI has the same number of zs
        if len(z_value_array) % n_roi != 0:
            msg += "There do not appear to be an "
            msg += "equal number of zs per ROI\n"
            msg += f"n_z: {len(z_value_array)} "
            msg += f"n_roi: {len(z_int_per_roi)}\n"

        for roi_z_ints in z_int_per_roi:
            these_z_ints = set([self._int_from_z(z_value=zz)
                                for zz in
                                z_value_array[offset:offset+n_z_per_roi]])
            if these_z_ints != roi_z_ints:
                these_z_ints = set([self._int_from_z(z_value=zz)
                                    for zz in
                                    z_value_array[offset:
                                                  offset+n_z_per_roi-1]])

                # might be placeholder value == 0
                odd_value = z_value_array[offset+n_z_per_roi-1]

                if np.abs(odd_value) >= 1.0e-6 or these_z_ints != roi_z_ints:
                    msg += "z_values from sub array "
                    msg += "not in correct order for ROIs; "
                    break
            offset += n_z_per_roi

        if len(msg) > 0:
            full_msg = "Unclear how to split this TIFF\n"
            full_msg += f"{self._file_path.resolve().absolute()}\n"
            full_msg += f"{msg}"
            full_msg += f"all_zs {self._metadata.all_zs()}\nfrom rois:\n"
            for roi in self._metadata.defined_rois:
                full_msg += f"zs: {roi['zs']}\n"
            raise RuntimeError(full_msg)

    def _get_z_manifest(self):
        """
        Populate various member objects that help us keep
        track of what z values go with what ROIs in this TIFF
        """
        local_z_value_list = np.array(self._metadata.all_zs()).flatten()
        defined_rois = self._metadata.defined_rois

        # create a list of sets indicating which z values were actually
        # scanned in the ROI (this will help us parse the placeholder
        # zeros that sometimes get dropped into
        # SI.hStackManager.zsAllActuators

        valid_z_int_per_roi = []
        valid_z_per_roi = []
        for roi in defined_rois:
            this_z_value = roi['zs']
            if isinstance(this_z_value, int):
                this_z_value = [this_z_value, ]
            z_as_int = [self._int_from_z(z_value=zz)
                        for zz in this_z_value]
            valid_z_int_per_roi.append(set(z_as_int))
            valid_z_per_roi.append(this_z_value)

        self._valid_z_int_per_roi = valid_z_int_per_roi
        self._valid_z_per_roi = valid_z_per_roi
        self._n_valid_zs = 0
        self._roi_z_int_manifest = []
        ct = 0
        i_roi = 0
        local_z_index_list = [self._int_from_z(zz)
                              for zz in local_z_value_list]
        for zz in local_z_index_list:
            if i_roi >= len(valid_z_int_per_roi):
                break
            if zz in valid_z_int_per_roi[i_roi]:
                roi_z = (i_roi, zz)
                self._roi_z_int_manifest.append(roi_z)
                self._n_valid_zs += 1
                ct += 1
                if ct == len(valid_z_int_per_roi[i_roi]):
                    i_roi += 1
                    ct = 0

    @property
    def input_path(self) -> pathlib.Path:
        """
        The file this splitter is trying to split
        """
        return self._file_path

    def is_z_valid_for_roi(self,
                           i_roi: int,
                           z_value: float) -> bool:
        """
        Is specified z-value valid for the specified ROI
        """
        z_as_int = self._int_from_z(z_value=z_value)
        return z_as_int in self._valid_z_int_per_roi[i_roi]

    @property
    def roi_z_int_manifest(self) -> List[Tuple[int, int]]:
        """
        A list of tuples. Each tuple is a valid
        (roi_index, z_as_int) pair.
        """
        return self._roi_z_int_manifest

    @property
    def n_valid_zs(self) -> int:
        """
        The total number of valid z values associated with this TIFF.
        """
        return self._n_valid_zs

    @property
    def n_rois(self) -> int:
        """
        The number of ROIs in this TIFF
        """
        return self._metadata.n_rois

    @property
    def n_pages(self):
        """
        The number of pages in this TIFF
        """
        if not hasattr(self, '_n_pages'):
            with tifffile.TiffFile(self._file_path, 'rb') as tiff_file:
                self._n_pages = len(tiff_file.pages)
        return self._n_pages

    def roi_center(self, i_roi: int) -> Tuple[float, float]:
        """
        The (X, Y) center coordinates of roi_index=i_roi
        """
        return self._metadata.roi_center(i_roi=i_roi)

    def _get_offset(self, i_roi: int, z_value: float) -> int:
        """
        Get the first page associated with the specified
        i_roi, z_value pair.
        """
        found_it = False
        n_step_over = 0
        this_roi_z = (i_roi, self._int_from_z(z_value=z_value))
        for roi_z_pair in self.roi_z_int_manifest:
            if roi_z_pair == this_roi_z:
                found_it = True
                break
            n_step_over += 1
        if not found_it:
            msg = f"Could not find stride for {i_roi}, {z_value}\n"
            msg += f"TIFF file {self._file_path.resolve().absolute()}"
            raise ValueError(msg)
        return n_step_over

    def _get_pages(self, i_roi: int, z_value: float) -> List[np.ndarray]:
        """
        Get a list of np.ndarrays representing the pages of image data
        for ROI i_roi at the specified z_value
        """

        if i_roi >= self.n_rois:
            msg = f"You asked for ROI {i_roi}; "
            msg += f"there are only {self.n_rois} ROIs "
            msg += f"in {self._file_path.resolve().absolute()}"
            raise ValueError(msg)

        if not self.is_z_valid_for_roi(i_roi=i_roi, z_value=z_value):
            msg = f"{z_value} is not a valid z value for ROI {i_roi};"
            msg += f"valid z values are {self._valid_z_per_roi[i_roi]}\n"
            msg += f"TIFF file {self._file_path.resolve().absolute()}"
            raise ValueError(msg)

        offset = self._get_offset(i_roi=i_roi, z_value=z_value)

        tiff_data = []
        with tifffile.TiffFile(self._file_path, 'rb') as tiff_file:
            for i_page in range(offset, self.n_pages, self.n_valid_zs):
                arr = tiff_file.pages[i_page].asarray()
                tiff_data.append(arr)
                key_pair = (i_roi, z_value)
                if key_pair in self._frame_shape:
                    if arr.shape != self._frame_shape[key_pair]:
                        msg = f"ROI {i_roi} z_value {z_value}\n"
                        msg += "yields inconsistent frame shape"
                        raise RuntimeError(msg)
                else:
                    self._frame_shape[key_pair] = arr.shape

        return tiff_data

    def frame_shape(self,
                    i_roi: int,
                    z_value: Optional[float]) -> Tuple[int, int]:
        """
        Get the shape of the image for a specified ROI at a specified
        z value

        Parameters
        ----------
        i_roi: int
            index of the ROI

        z_value: Optional[float]
            value of z. If None, z_value will be detected automaticall
            (assuming there is no ambiguity)

        Returns
        -------
        frame_shape: Tuple[int, int]
            (nrows, ncolumns)
        """
        if z_value is None:
            z_value = self._get_z_value(i_roi=i_roi)

        key_pair = (i_roi, self._int_from_z(z_value))

        if key_pair not in self._frame_shape:
            offset = self._get_offset(i_roi=i_roi, z_value=z_value)
            with tifffile.TiffFile(self._file_path, 'rb') as tiff_file:
                page = tiff_file.pages[offset].asarray()
                self._frame_shape[key_pair] = page.shape
        return self._frame_shape[key_pair]

    def _get_z_value(self, i_roi: int) -> float:
        """
        Return the z_value associated with i_roi, assuming
        there is only one. Raises a RuntimeError if there
        is more than one.
        """
        # When splitting surface TIFFs, there's no sensible
        # way to know the z-value ahead of time (whatever the
        # operator enters is just a placeholder). The block
        # of code below will scan for z-values than align with
        # the specified ROI ID and select the correct z value
        # (assuming there is only one)
        possible_z_values = []
        for pair in self.roi_z_int_manifest:
            if pair[0] == i_roi:
                possible_z_values.append(pair[1])
        if len(possible_z_values) > 1:
            msg = f"{len(possible_z_values)} possible z values "
            msg += f"for ROI {i_roi}; must specify one of\n"
            msg += f"{possible_z_values}"
            raise RuntimeError(msg)
        z_value = possible_z_values[0]
        return self._z_from_int(ii=z_value)

    def write_output_file(self,
                          i_roi: int,
                          z_value: Optional[float],
                          output_path: pathlib.Path) -> None:
        """
        Write the image created by averaging all of the TIFF
        pages associated with an (i_roi, z_value) pair to a TIFF
        file.

        Parameters
        ----------
        i_roi: int

        z_value: Optional[int]
            If None, will be detected automatically (assuming there
            is only one)

        output_path: pathlib.Path
            Path to file to be written

        Returns
        -------
        None
            Output is written to output_path
        """

        if output_path.suffix not in ('.tif', '.tiff'):
            msg = "expected .tiff output path; "
            msg += f"you specified {output_path.resolve().absolute()}"

        if z_value is None:
            z_value = self._get_z_value(i_roi=i_roi)
        data = np.array(self._get_pages(i_roi=i_roi, z_value=z_value))
        avg_img = np.mean(data, axis=0)
        avg_img = normalize_array(array=avg_img,
                                  lower_cutoff=None,
                                  upper_cutoff=None)
        tifffile.imsave(output_path, avg_img)
        return None


class TimeSeriesSplitter(ScanImageTiffSplitter):
    """
    A class specifically for splitting timeseries TIFFs

    Parameters
    ----------
    tiff_path: pathlib.Path
        Path to the TIFF file whose metadata we are parsing
    """

    def _get_pages(self, i_roi: int, z_value: float) -> None:
        msg = "_get_pages is not implemented for "
        msg += "TimeSeriesSplitter; the resulting "
        msg += "output would be unreasonably large. "
        msg += "Call write_video_h5 instead to write "
        msg += "data directly to an HDF5 file."
        raise NotImplementedError(msg)

    def write_output_file(self,
                          i_roi: int,
                          z_value: float,
                          output_path: pathlib.Path) -> None:
        """
        Write all of the pages associated with an
        (i_roi, z_value) pair to an HDF5 file.

        Parameters
        ----------
        i_roi: int
            index of the ROI

        z_value: int
            z value of the scanfield plane

        output_path: pathlib.Path
            path to the HDF5 file to be written

        Returns
        -------
        None
            output is written to output_path
        """

        if output_path.suffix != '.h5':
            msg = "expected HDF5 output path; "
            msg += f"you gave {output_path.resolve().absolute()}"
            raise ValueError(msg)

        if i_roi >= self.n_rois:
            msg = f"You asked for ROI {i_roi}; "
            msg += f"there are only {self.n_rois} ROIs "
            msg += f"in {self._file_path.resolve().absolute()}"
            raise ValueError(msg)

        if not self.is_z_valid_for_roi(i_roi=i_roi, z_value=z_value):
            msg = f"{z_value} is not a valid z value for ROI {i_roi};"
            msg += f"valid z values are {self._valid_z_per_roi[i_roi]}\n"
            msg += f"TIFF file {self._file_path.resolve().absolute()}"
            raise ValueError(msg)

        offset = self._get_offset(i_roi=i_roi, z_value=z_value)

        n_frames = np.ceil((self.n_pages-offset)/self.n_valid_zs).astype(int)

        with tifffile.TiffFile(self._file_path, 'rb') as tiff_file:
            eg_array = tiff_file.pages[0].asarray()
            fov_shape = eg_array.shape
            video_dtype = eg_array.dtype

            with h5py.File(output_path, 'w') as out_file:
                out_file.create_dataset(
                            'data',
                            shape=(n_frames, fov_shape[0], fov_shape[1]),
                            dtype=video_dtype,
                            chunks=(max(1, n_frames//1000),
                                    fov_shape[0], fov_shape[1]))

                for i_frame, i_page in enumerate(range(offset,
                                                       self.n_pages,
                                                       self.n_valid_zs)):
                    arr = tiff_file.pages[i_page].asarray()
                    out_file['data'][i_frame, :, :] = arr

        return None