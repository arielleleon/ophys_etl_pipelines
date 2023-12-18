import json
import h5py
import numpy as np
from argschema import ArgSchemaParser
from marshmallow import ValidationError
import json
from pathlib import Path
from aind_data_schema.core.processing import Processing, DataProcess, ProcessName, PipelineProcess
from typing import Union
from datetime import datetime as dt
from datetime import timezone as tz
from pathlib import Path

from ophys_etl.utils.motion_border import (
        get_max_correction_from_file,
        MaxFrameShift)
from ophys_etl.schemas import DenseROISchema
from ophys_etl.utils.rois import (binarize_roi_mask,
                                  coo_rois_to_lims_compatible,
                                  suite2p_rois_to_coo,
                                  morphological_transform)
from ophys_etl.modules.postprocess_rois.utils import filter_by_aspect_ratio
from ophys_etl.modules.postprocess_rois.schemas import \
        PostProcessROIsInputSchema

def write_output_metadata(
    metadata: dict,
    input_fp: Union[str, Path],
    output_fp: Union[str, Path],
    url: str,
    start_date_time: dt,
) -> None:
    """Writes output metadata to plane processing.json

    Parameters
    ----------
    metadata: dict
        parameters from suite2p motion correction
    input_fp: str
        path to data input
    output_fp: str
        path to data output
    url: str
        url to code repository
    """
    processing = Processing(
        processing_pipeline=PipelineProcess(
            processor_full_name="Multplane Ophys Processing Pipeline",
            pipeline_url="https://codeocean.allenneuraldynamics.org/capsule/5472403/tree",
            pipeline_version="0.1.0",
            data_processes=[
                DataProcess(
                    name=ProcessName.VIDEO_ROI_SEGMENTATION,
                    software_version="0.1.0",
                    start_date_time=start_date_time,  # TODO: Add actual dt
                    end_date_time=dt.now(tz.utc),  # TODO: Add actual dt
                    input_location=str(input_fp),
                    output_location=str(output_fp),
                    code_url=(url),
                    parameters=metadata,
                )
            ],
        )
    )
    print(f"Output filepath: {output_fp}")
    with open(Path(output_fp).parent.parent.parent / "processing.json", "r") as f:
        proc_data = json.load(f)
    processing.write_standard_file(output_directory=Path(output_fp).parent.parent.parent)
    with open(Path(output_fp).parent.parent.parent / "processing.json", "r") as f:
        dct_data = json.load(f)
    proc_data["processing_pipeline"]["data_processes"].append(
        dct_data["processing_pipeline"]["data_processes"][0]
    )
    with open(Path(output_fp).parent.parent.parent / "processing.json", "w") as f:
        json.dump(proc_data, f, indent=4)


class PostProcessROIs(ArgSchemaParser):
    default_schema = PostProcessROIsInputSchema

    def run(self):
        """
        This function takes ROIs (regions of interest) outputted from
        suite2p in a stat.npy file and converts them to a LIMS compatible
        data format for storage and further processing. This process
        binarizes the masks and then changes the formatting before writing
        to a json output file.
        """
        start_time = dt.now(tz.utc)
        self.logger.name = type(self).__name__
        self.logger.setLevel(self.args.pop('log_level'))

        # load in the rois from the stat file and movie path for shape
        self.logger.info("Loading suite2p ROIs and video size.")
        suite2p_stats = np.load(self.args['suite2p_stat_path'],
                                allow_pickle=True)
        with h5py.File(self.args['motion_corrected_video'], 'r') as open_vid:
            movie_shape = open_vid['data'][0].shape

        # binarize the masks
        self.logger.info("Filtering and Binarizing the ROIs created by "
                         "Suite2p.")
        coo_rois = suite2p_rois_to_coo(suite2p_stats, movie_shape)

        # filter raw rois by aspect ratio
        filtered_coo_rois = filter_by_aspect_ratio(
                coo_rois,
                self.args['aspect_ratio_threshold'])
        self.logger.info("Filtered out "
                         f"{len(coo_rois) - len(filtered_coo_rois)} "
                         "ROIs with aspect ratio <= "
                         f"{self.args['aspect_ratio_threshold']}")

        binarized_coo_rois = []
        for filtered_coo_roi in filtered_coo_rois:
            binary_mask = binarize_roi_mask(filtered_coo_roi,
                                            self.args['abs_threshold'],
                                            self.args['binary_quantile'])
            binarized_coo_rois.append(binary_mask)
        self.logger.info("Binarized ROIs from Suite2p, total binarized: "
                         f"{len(binarized_coo_rois)}")

        # load the motion correction values
        self.logger.info("Loading motion correction border values from "
                         f" {self.args['motion_correction_values']}")

        if self.args['motion_correction_values'] is not None:
            max_frame_shift = get_max_correction_from_file(
                self.args['motion_correction_values'],
                self.args['maximum_motion_shift'])
        else:
            max_frame_shift = MaxFrameShift(left=0, right=0, up=0, down=0)

        # create the rois
        self.logger.info("Transforming ROIs to LIMS compatible style.")
        compatible_rois = coo_rois_to_lims_compatible(
                binarized_coo_rois, max_frame_shift, movie_shape,
                self.args['npixel_threshold'])

        if self.args['morphological_ops']:
            compatible_rois = [morphological_transform(roi, shape=movie_shape)
                               for roi in compatible_rois]
            n_rois = len(compatible_rois)
            # eliminate None
            compatible_rois = [roi for roi in compatible_rois if roi]
            n_rois_morphed = len(compatible_rois)
            self.logger.info("morphological transform reduced number of "
                             f"ROIs from {n_rois} to {n_rois_morphed}")

        # validate ROIs
        errors = DenseROISchema(many=True).validate(compatible_rois)
        if any(errors):
            raise ValidationError(f"Schema validation errors: {errors}")

        # save the rois as a json file to output directory
        self.logger.info("Writing LIMs compatible ROIs to json file at "
                         f"{self.args['output_json']}")

        url = "https://github.com/AllenNeuralDynamics/aind-ophys-segmentation-cellpose"
        write_output_metadata(
            {'compatible_rois': compatible_rois},
            self.args['motion_corrected_video'],
            self.args['output_json'],
            url,
            start_time
        )
        with open(self.args['output_json'], 'w') as f:
            json.dump(compatible_rois, f, indent=2)


if __name__ == '__main__':  # pragma: no cover
    roi_post = PostProcessROIs()
    roi_post.run()
