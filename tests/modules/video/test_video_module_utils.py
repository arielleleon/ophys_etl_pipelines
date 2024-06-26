import pytest
import numpy as np
import h5py
import tempfile
import pathlib
import hashlib
from itertools import product
from functools import partial

from ophys_etl.utils.array_utils import (
    downsample_array,
    n_frames_from_hz)

from ophys_etl.modules.median_filtered_max_projection.utils import (
    apply_median_filter_to_video)

from ophys_etl.modules.video.utils import (
    apply_downsampled_mean_filter_to_video,
    _video_worker,
    create_downsampled_video_h5,
    _write_array_to_video,
    _min_max_from_h5,
    _video_array_from_h5,
    create_downsampled_video,
    create_side_by_side_video,
    add_reticle,
    _get_post_filter_frame_size)


class DummyContextManager(object):
    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        return


def test_get_post_filter_frame_size():
    video = np.zeros((10, 23, 22), dtype=int)
    assert _get_post_filter_frame_size(
               example_video=video,
               spatial_filter=None) == (23, 22)

    def silly_filter(input_video):
        return np.zeros((input_video.shape[0], 3, 4), dtype=float)

    assert _get_post_filter_frame_size(
                example_video=video,
                spatial_filter=silly_filter) == (3, 4)


@pytest.mark.parametrize(
    "video_dtype, d_reticle",
    product([np.uint8, np.uint16], [6, 7]))
def test_add_reticle(video_dtype, d_reticle):
    max_val = np.iinfo(video_dtype).max
    video_data = (max_val//2)*np.ones((10, 40, 43, 3),
                                      dtype=video_dtype)

    input_data = np.copy(video_data)

    video_data = add_reticle(video_array=video_data,
                             d_reticle=d_reticle)

    assert not np.array_equal(input_data, video_data)

    expected_unchanged = max_val//2
    expected_just_one = (3*(expected_unchanged//4)) + (max_val//4)
    expected_overlap = (3*(expected_just_one//4)) + (max_val//4)

    for iy in range(video_data.shape[1]):
        is_row_grid = False
        if iy % d_reticle < 2 and iy > 1:
            is_row_grid = True
        for ix in range(video_data.shape[2]):
            is_col_grid = False
            if ix % d_reticle < 2 and ix > 1:
                is_col_grid = True

            if not is_row_grid and not is_col_grid:
                assert (video_data[:, iy, ix, :] == expected_unchanged).all()
            elif is_row_grid and is_col_grid:
                assert (video_data[:, iy, ix, 0] == expected_overlap).all()
            else:
                assert (video_data[:, iy, ix, 0] == expected_just_one).all()


@pytest.mark.parametrize(
    "kernel_size, nrows, ncols",
    product((2, 4, 7), (32, 13), (32, 23)))
def test_apply_downsampled_mean_filter_to_video(
        kernel_size,
        nrows,
        ncols):
    rng = np.random.default_rng(235813)
    video_data = rng.random((22, nrows, ncols))
    ds_video = apply_downsampled_mean_filter_to_video(
                    video=video_data,
                    kernel_size=kernel_size)

    expected_shape = (video_data.shape[0],
                      np.ceil(video_data.shape[1]/kernel_size).astype(int),
                      np.ceil(video_data.shape[2]/kernel_size).astype(int))

    assert ds_video.shape == expected_shape

    for i_time in range(video_data.shape[0]):
        for iy in range(0, nrows, kernel_size):
            for ix in range(0, ncols, kernel_size):
                chunk = video_data[i_time,
                                   iy:iy+kernel_size,
                                   ix:ix+kernel_size]
                expected = chunk.sum()/(kernel_size**2)
                iy_ds = np.ceil(iy/kernel_size).astype(int)
                ix_ds = np.ceil(ix/kernel_size).astype(int)
                actual = ds_video[i_time, iy_ds, ix_ds]
                np.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize(
    "output_hz, input_slice, kernel_size",
    product((12.0, 6.0, 4.0),
            ((6, 36), (12, 42), (30, 53)),
            (None, 2, 3)))
def test_video_module_worker(
        tmpdir,
        video_path_fixture,
        video_array_fixture,
        output_hz,
        input_slice,
        kernel_size):
    """
    Test that _video_worker writes the expected result to the output file
    """

    input_hz = 12.0

    frames_to_group = n_frames_from_hz(
            input_hz,
            output_hz)

    # find the non-zero indices of the output file
    output_start = input_slice[0] // frames_to_group
    d_slice = input_slice[1] - input_slice[0]
    output_end = output_start + np.ceil(d_slice/frames_to_group).astype(int)

    output_path = tempfile.mkstemp(dir=tmpdir,
                                   prefix='video_worker_test_',
                                   suffix='.h5')[1]
    output_path = pathlib.Path(output_path)

    with h5py.File(output_path, 'w') as out_file:
        dummy_data = np.zeros(video_array_fixture.shape,
                              dtype=video_array_fixture.dtype)
        out_file.create_dataset('data',
                                data=dummy_data)

    this_slice = video_array_fixture[input_slice[0]:input_slice[1], :, :]

    if output_hz < input_hz:
        expected = downsample_array(this_slice,
                                    input_fps=input_hz,
                                    output_fps=output_hz,
                                    strategy='average')
    else:
        expected = np.copy(this_slice)

    if kernel_size is not None:
        expected = apply_median_filter_to_video(expected, kernel_size)
        spatial_filter = partial(apply_median_filter_to_video,
                                 kernel_size=kernel_size)
    else:
        spatial_filter = None

    lock = DummyContextManager()
    _video_worker(
            video_path_fixture,
            input_hz,
            output_path,
            output_hz,
            spatial_filter,
            input_slice,
            dict(),
            lock)

    with h5py.File(output_path, 'r') as in_file:
        full_data = in_file['data'][()]
    actual = full_data[output_start:output_end, :, :]
    np.testing.assert_array_equal(actual, expected)

    # make sure other pixels in output file were not touched
    other_mask = np.ones(full_data.shape, dtype=bool)
    other_mask[output_start:output_end, :, :] = False
    other_values = np.unique(full_data[other_mask])
    assert len(other_values) == 1
    assert np.abs(other_values[0]) < 1.0e-20


def test_video_module_worker_exception(
        video_path_fixture):
    """
    Test that exception is raised by _video_worker when input_slice[0]
    is not an integer multiple of the chunk size of frames used in
    downsampling
    """
    input_hz = 12.0
    output_hz = 6.0
    input_slice = [5, 19]
    spatial_filter = partial(apply_median_filter_to_video,
                             kernel_size=3)
    output_path = pathlib.Path('silly.h5')

    with pytest.raises(RuntimeError, match="integer multiple"):
        lock = DummyContextManager()
        validity_dict = dict()
        _video_worker(
                video_path_fixture,
                input_hz,
                output_path,
                output_hz,
                spatial_filter,
                input_slice,
                validity_dict,
                lock)
    assert len(validity_dict) == 1
    for k in validity_dict:
        assert not validity_dict[k][0]
        assert "integer multiple" in validity_dict[k][1]


@pytest.mark.parametrize(
    "output_hz, kernel_size",
    product((12.0, 5.0), (None, 3)))
def test_module_create_video_h5(
        tmpdir,
        video_path_fixture,
        output_hz,
        kernel_size):
    """
    This is really just a smoke test
    """
    if kernel_size is not None:
        spatial_filter = partial(apply_median_filter_to_video,
                                 kernel_size=kernel_size)
    else:
        spatial_filter = None

    output_path = pathlib.Path(tempfile.mkstemp(
                                   dir=tmpdir,
                                   prefix="create_ideo_smoke_test_",
                                   suffix=".h5")[1])
    create_downsampled_video_h5(
        video_path_fixture,
        12.0,
        output_path,
        output_hz,
        spatial_filter,
        3)


@pytest.mark.parametrize(
    "video_suffix, fps, quality",
    product((".mp4", ".avi", ".tiff", ".tif"),
            (5, 10),
            (3, 5, 8)))
def test_module_write_array_to_video(
        tmpdir,
        video_array_fixture,
        video_suffix,
        fps,
        quality):
    """
    This is just a smoke test of code that calls
    imageio to write the video files.
    """

    video_path = pathlib.Path(
                     tempfile.mkstemp(dir=tmpdir,
                                      prefix="dummy_",
                                      suffix=video_suffix)[1])

    _write_array_to_video(
        video_path,
        video_array_fixture,
        fps,
        quality)

    assert video_path.is_file()


@pytest.mark.parametrize("border", (1, 2, 3, 100))
def test_min_max_from_h5_no_quantiles(
        video_path_fixture,
        video_array_fixture,
        border):

    nrows = video_array_fixture.shape[1]
    ncols = video_array_fixture.shape[2]

    if border < 8:
        this_array = video_array_fixture[:,
                                         border:nrows-border,
                                         border:ncols-border]
    else:
        this_array = np.copy(video_array_fixture)

    expected_min = this_array.min()
    expected_max = this_array.max()

    actual = _min_max_from_h5(
                    h5_path=video_path_fixture,
                    quantiles=(0.0, 1.0),
                    border=border)

    assert np.abs(actual[0]-expected_min) < 1.0e-20
    assert np.abs(actual[1]-expected_max) < 1.0e-20


@pytest.mark.parametrize("border, quantiles",
                         product((1, 2, 3), ((0.1, 0.9), (0.2, 0.8))))
def test_min_max_from_h5_with_quantiles(
        video_path_fixture,
        video_array_fixture,
        border,
        quantiles):

    nrows = video_array_fixture.shape[1]
    ncols = video_array_fixture.shape[2]
    this_array = video_array_fixture[:,
                                     border:nrows-border,
                                     border:ncols-border]

    (expected_min,
     expected_max) = np.quantile(this_array, quantiles)

    actual = _min_max_from_h5(
                    video_path_fixture,
                    quantiles,
                    border)

    assert np.abs(actual[0]-expected_min) < 1.0e-20
    assert np.abs(actual[1]-expected_max) < 1.0e-20


@pytest.mark.parametrize(
    "min_val, max_val, video_dtype",
    product((50.0, 100.0, 250.0),
            (1900.0, 1500.0, 1000.0),
            (np.uint8, np.uint16)))
def test_module_video_array_from_h5_no_reticle(
        video_path_fixture,
        video_array_fixture,
        min_val,
        max_val,
        video_dtype):

    if video_dtype == np.uint8:
        max_cast = 255
    else:
        max_cast = 65535

    video_array = _video_array_from_h5(
                        video_path_fixture,
                        min_val=min_val,
                        max_val=max_val,
                        reticle=False,
                        video_dtype=video_dtype)

    assert len(video_array.shape) == 4
    assert video_array.shape == (video_array_fixture.shape[0],
                                 video_array_fixture.shape[1],
                                 video_array_fixture.shape[2],
                                 3)

    below_min = np.where(video_array_fixture < min_val)
    assert len(below_min[0]) > 0
    assert (video_array[below_min] == 0).all()
    above_max = np.where(video_array_fixture > max_val)
    assert len(above_max[0]) > 0
    assert (video_array[above_max] == max_cast).all()
    assert video_array.min() == 0
    assert video_array.max() == max_cast
    assert video_array.dtype == video_dtype


def test_module_video_array_from_h5_exception(
        video_path_fixture):

    with pytest.raises(ValueError, match="either np.uint8 or np.uint16"):
        _ = _video_array_from_h5(
                        video_path_fixture,
                        min_val=0.0,
                        max_val=100.0,
                        reticle=False,
                        video_dtype=float)


@pytest.mark.parametrize("d_reticle, video_dtype",
                         product((5, 7, 9), (np.uint8, np.uint16)))
def test_video_array_from_h5_with_reticle(
        video_path_fixture,
        video_array_fixture,
        d_reticle,
        video_dtype):

    min_val = 500.0
    max_val = 1500.0
    video_shape = video_array_fixture.shape

    no_reticle = _video_array_from_h5(
                        video_path_fixture,
                        min_val=min_val,
                        max_val=max_val,
                        reticle=False,
                        d_reticle=d_reticle,
                        video_dtype=video_dtype)

    yes_reticle = _video_array_from_h5(
                        video_path_fixture,
                        min_val=min_val,
                        max_val=max_val,
                        reticle=True,
                        d_reticle=d_reticle,
                        video_dtype=video_dtype)

    reticle_mask = np.zeros(no_reticle.shape, dtype=bool)
    for ii in range(d_reticle, video_shape[1], d_reticle):
        reticle_mask[:, ii:ii+2, :, :] = True
    for ii in range(d_reticle, video_shape[2], d_reticle):
        reticle_mask[:, :, ii:ii+2, :] = True

    assert reticle_mask.sum() > 0

    np.testing.assert_array_equal(
            no_reticle[np.logical_not(reticle_mask)],
            yes_reticle[np.logical_not(reticle_mask)])

    assert not np.array_equal(no_reticle[reticle_mask],
                              yes_reticle[reticle_mask])


@pytest.mark.parametrize(
    "output_suffix, kernel_size, reticle",
    product((".avi", ".mp4", ".tiff"),
            (None, 5),
            (True, False)))
def test_module_create_downsampled_video(
        tmpdir,
        video_path_fixture,
        video_array_fixture,
        output_suffix,
        kernel_size,
        reticle):
    """
    This will test create_downsampled_video by calling all of the
    constituent parts by hand and verifying that the md5checksum
    of the file produced that way matches the md5checksum of the file
    produced by calling create_downsampled_video. It's a little
    tautological, but it will tell us if any part of our pipeline is
    no longer self-consistent.
    """

    quantiles = (0.3, 0.9)
    input_hz = 12.0
    output_hz = 5.0
    speed_up_factor = 3
    quality = 4
    d_reticle = 64  # because we haven't exposed this to the user, yet
    expected_file = pathlib.Path(
                        tempfile.mkstemp(dir=tmpdir,
                                         prefix='video_expected_',
                                         suffix=output_suffix)[1])

    downsampled_video = downsample_array(
                            video_array_fixture,
                            input_fps=input_hz,
                            output_fps=output_hz,
                            strategy='average')

    if kernel_size is not None:
        downsampled_video = apply_median_filter_to_video(
                                    downsampled_video,
                                    kernel_size)
        spatial_filter = partial(apply_median_filter_to_video,
                                 kernel_size=kernel_size)
    else:
        spatial_filter = None

    (min_val,
     max_val) = np.quantile(downsampled_video, quantiles)

    downsampled_video = downsampled_video.astype(float)
    downsampled_video = np.where(downsampled_video > min_val,
                                 downsampled_video, min_val)
    downsampled_video = np.where(downsampled_video < max_val,
                                 downsampled_video, max_val)

    delta = max_val-min_val
    downsampled_video = np.round(255.0*(downsampled_video-min_val)/delta)
    video_as_uint = np.zeros((downsampled_video.shape[0],
                              downsampled_video.shape[1],
                              downsampled_video.shape[2],
                              3), dtype=np.uint8)

    for ic in range(3):
        video_as_uint[:, :, :, ic] = downsampled_video

    video_shape = video_as_uint.shape
    del downsampled_video

    if reticle:
        for ii in range(d_reticle, video_shape[1], d_reticle):
            old_vals = np.copy(video_as_uint[:, ii:ii+2, :, :])
            new_vals = np.zeros(old_vals.shape, dtype=np.uint8)
            new_vals[:, :, :, 0] = 255
            new_vals = (new_vals//2) + (old_vals//2)
            new_vals = new_vals.astype(np.uint8)
            video_as_uint[:, ii:ii+2, :, :] = new_vals
        for ii in range(d_reticle, video_shape[2], d_reticle):
            old_vals = np.copy(video_as_uint[:, :, ii:ii+2, :])
            new_vals = np.zeros(old_vals.shape, dtype=np.uint8)
            new_vals[:, :, :, 0] = 255
            new_vals = (new_vals//2) + (old_vals//2)
            new_vals = new_vals.astype(np.uint8)
            video_as_uint[:, :, ii:ii+2, :] = new_vals

    _write_array_to_video(
            expected_file,
            video_as_uint,
            int(speed_up_factor*output_hz),
            quality)

    assert expected_file.is_file()

    actual_file = pathlib.Path(
                        tempfile.mkstemp(dir=tmpdir,
                                         prefix='video_actual_',
                                         suffix=output_suffix)[1])

    create_downsampled_video(
            video_path_fixture,
            input_hz,
            actual_file,
            output_hz,
            spatial_filter,
            3,
            quality=quality,
            quantiles=quantiles,
            reticle=reticle,
            speed_up_factor=speed_up_factor,
            tmp_dir=tmpdir)

    assert actual_file.is_file()

    md5_expected = hashlib.md5()
    with open(expected_file, 'rb') as in_file:
        chunk = in_file.read(100000)
        while len(chunk) > 0:
            md5_expected.update(chunk)
            chunk = in_file.read(100000)

    md5_actual = hashlib.md5()
    with open(actual_file, 'rb') as in_file:
        chunk = in_file.read(100000)
        while len(chunk) > 0:
            md5_actual.update(chunk)
            chunk = in_file.read(100000)

    assert md5_actual.hexdigest() == md5_expected.hexdigest()


@pytest.mark.parametrize(
    "output_suffix, output_hz, kernel_size, reticle",
    product((".avi", ".mp4", ".tiff"),
            (3.0, 5.0),
            (None, 5),
            (True, False)))
def test_module_create_side_by_side_video(
        tmpdir,
        video_path_fixture,
        video_array_fixture,
        output_suffix,
        output_hz,
        kernel_size,
        reticle):
    """
    This is just going to be a smoke test, as it's hard to verify
    the contents of an mp4
    """

    quantiles = (0.3, 0.9)
    quality = 4
    speed_up_factor = 2

    actual_file = pathlib.Path(
                        tempfile.mkstemp(dir=tmpdir,
                                         prefix='side_by_side_actual_',
                                         suffix=output_suffix)[1])

    if kernel_size is not None:
        spatial_filter = partial(apply_median_filter_to_video,
                                 kernel_size=kernel_size)
    else:
        spatial_filter = None

    input_hz = 12.0
    create_side_by_side_video(
            video_path_fixture,
            video_path_fixture,
            input_hz,
            actual_file,
            output_hz,
            spatial_filter,
            3,
            quality,
            quantiles,
            reticle,
            speed_up_factor,
            tmpdir)

    assert actual_file.is_file()


def test_module_create_side_by_side_video_shape_error(
        tmpdir,
        video_path_fixture):
    """
    Test that create_side_by_side_video raises an error when the
    two videos have different shapes
    """
    out_path = tempfile.mkstemp(dir=tmpdir, suffix='.avi')[1]
    out_path = pathlib.Path(out_path)

    other_path = tempfile.mkstemp(dir=tmpdir, suffix='.h5')[1]
    other_path = pathlib.Path(other_path)
    with h5py.File(other_path, 'w') as out_file:
        out_file.create_dataset('data', data=np.zeros((2, 2, 2, 2)))

    with pytest.raises(RuntimeError, match='Videos need to be the same shape'):
        create_side_by_side_video(
                left_video_path=video_path_fixture,
                right_video_path=other_path,
                input_hz=5.0,
                output_path=out_path,
                output_hz=1.0,
                spatial_filter=None,
                n_processors=3,
                quality=5,
                quantiles=(0.0, 1.0),
                reticle=False,
                speed_up_factor=2,
                tmp_dir=tmpdir)
