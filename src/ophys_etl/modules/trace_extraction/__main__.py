import argschema

from ophys_etl.modules.trace_extraction.utils import extract_traces
from ophys_etl.modules.trace_extraction.schemas import (
        TraceExtractionInputSchema, TraceExtractionOutputSchema)
import json
from pathlib import Path
from aind_data_schema.core.processing import Processing, DataProcess, ProcessName, PipelineProcess
from typing import Union
from datetime import datetime as dt
from datetime import timezone as tz

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
        with open(metadata['neuropil_mask_file'], "r") as f:
                metadata = json.load(f)
        processing = Processing(
                processing_pipeline=PipelineProcess(
                processor_full_name="Multplane Ophys Processing Pipeline",
                pipeline_url="https://codeocean.allenneuraldynamics.org/capsule/5472403/tree",
                pipeline_version="0.1.0",
                data_processes=[
                        DataProcess(
                        name=ProcessName.VIDEO_ROI_TIMESERIES_EXTRACTION,
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

class TraceExtraction(argschema.ArgSchemaParser):
    default_schema = TraceExtractionInputSchema
    default_output_schema = TraceExtractionOutputSchema

    def run(self):
        start_time = dt.now(tz.utc)
        self.logger.name = type(self).__name__
        output = extract_traces(
                self.args['motion_corrected_stack'],
                self.args['motion_border'],
                self.args['storage_directory'],
                self.args['rois'])
        self.output(output, indent=2)
        write_output_metadata(
                metadata=output,
                input_fp = output['neuropil_mask_file'],
                output_fp = output['neuropil_mask_file'],
                url = "https://github.com/AllenNeuralDynamics/aind-ophys-trace-extraction.git",
                start_date_time = start_time
        )
        

if __name__ == '__main__':  # pragma: nocover
    te = TraceExtraction()
    te.run()
