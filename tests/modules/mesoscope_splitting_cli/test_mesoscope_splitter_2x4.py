# This file defines all of the independent data fixture necessary
# to test TIFF splitting in the case of a 2 ROI x 4 plane OPhys session

import pytest
import pathlib
import copy
from itertools import product
from utils import run_mesoscope_cli_test


@pytest.fixture
def float_resolution_fixture():
    return 0


@pytest.fixture
def splitter_tmp_dir_fixture(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp('splitter_cli_test')
    return pathlib.Path(tmp_dir)


@pytest.fixture
def z_to_exp_id_fixture():
    """
    dict mapping z value to ophys_experiment_id
    """
    return {(2, 1): {2: 222, 1: 111},
            (3, 4): {3: 333, 4: 444},
            (6, 5): {6: 666, 5: 555},
            (8, 7): {8: 888, 7: 777}}


@pytest.fixture
def z_list_fixture():
    z_list = [[2, 1], [3, 4], [6, 5], [8, 7]]
    return z_list


@pytest.fixture
def roi_index_to_z_fixture():
    return {0: [1, 2, 3, 4], 1: [5, 6, 7, 8]}


@pytest.fixture
def z_to_roi_index_fixture():
    return {(2, 1): 0,
            (3, 4): 0,
            (6, 5): 1,
            (8, 7): 1}


@pytest.mark.parametrize(
        'use_platform_json, use_data_upload_dir',
        product((True, False, None), (True, False, None)))
def test_splitter_2x4(input_json_fixture,
                      tmp_path_factory,
                      zstack_metadata_fixture,
                      surface_metadata_fixture,
                      image_metadata_fixture,
                      depth_fixture,
                      surface_fixture,
                      timeseries_fixture,
                      zstack_fixture,
                      full_field_2p_tiff_fixture,
                      use_platform_json,
                      use_data_upload_dir):

    input_json_data = copy.deepcopy(input_json_fixture)
    expect_full_field = True
    if use_platform_json is None:
        input_json_data['platform_json_path'] = None
        expect_full_field = False
    elif not use_platform_json:
        input_json_data.pop('platform_json_path')
        expect_full_field = False

    if use_data_upload_dir is None:
        input_json_data['data_upload_dir'] = None
        expect_full_field = False
    elif not use_data_upload_dir:
        input_json_data.pop('data_upload_dir')
        expect_full_field = False

    tmp_dir = tmp_path_factory.mktemp('cli_output_json_2x4')
    tmp_dir = pathlib.Path(tmp_dir)
    exp_ct = run_mesoscope_cli_test(
                input_json=input_json_data,
                tmp_dir=tmp_dir,
                zstack_metadata=zstack_metadata_fixture,
                surface_metadata=surface_metadata_fixture,
                image_metadata=image_metadata_fixture,
                depth_data=depth_fixture,
                surface_data=surface_fixture,
                timeseries_data=timeseries_fixture,
                zstack_data=zstack_fixture,
                full_field_2p_tiff_data=full_field_2p_tiff_fixture,
                expect_full_field=expect_full_field)

    assert exp_ct == 8
