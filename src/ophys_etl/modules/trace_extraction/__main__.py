import argschema

from ophys_etl.modules.trace_extraction.utils import extract_traces
from ophys_etl.modules.trace_extraction.schemas import (
        TraceExtractionInputSchema, TraceExtractionOutputSchema)
from aind_data_schema.core.processing import ProcessName
from datetime import datetime as dt
from datetime import timezone as tz

from ophys_etl.utils.process_metadata import write_output_metadata

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
                output,
                ProcessName.TRACE_EXTRACTION,
                input_fp = output['neuropil_mask_file'],
                output_fp = output['neuropil_mask_file'],
                start_date_time = start_time
        )
        

if __name__ == '__main__':  # pragma: nocover
    te = TraceExtraction()
    te.run()
