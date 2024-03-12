from pathlib import Path
from aind_data_schema.core.processing import Processing, DataProcess, PipelineProcess
from typing import Union
from datetime import datetime as dt
from datetime import timezone as tz
from pathlib import Path
import os
import json


def write_output_metadata(
    metadata: dict,
    process_name: str,
    input_fp: Union[str, Path],
    output_fp: Union[str, Path],
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
    """
    processing = Processing(
        processing_pipeline=PipelineProcess(
            processor_full_name="Multplane Ophys Processing Pipeline",
            pipeline_url="https://codeocean.allenneuraldynamics.org/capsule/5472403/tree",
            pipeline_version="0.1.0",
            data_processes=[
                DataProcess(
                    name=process_name,
                    software_version=os.getenv("OPHYS_ETL_COMMIT_SHA"),
                    start_date_time=start_date_time,  # TODO: Add actual dt
                    end_date_time=dt.now(tz.utc),  # TODO: Add actual dt
                    input_location=str(input_fp),
                    output_location=str(output_fp),
                    code_url=(os.getenv("OPHYS_ETL_URL")),
                    parameters=metadata,
                )
            ],
        )
    )
    print(f"Output filepath: {output_fp}")
    with open(Path(output_fp).parent.parent / "processing.json", "r") as f:
        proc_data = json.load(f)
    processing.write_standard_file(output_directory=Path(output_fp).parent.parent)
    with open(Path(output_fp).parent.parent / "processing.json", "r") as f:
        dct_data = json.load(f)
    proc_data["processing_pipeline"]["data_processes"].append(
        dct_data["processing_pipeline"]["data_processes"][0]
    )
    with open(Path(output_fp).parent.parent / "processing.json", "w") as f:
        json.dump(proc_data, f, indent=4)