import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

def run():
    pipeline_options = PipelineOptions()
    
    with beam.Pipeline(options=pipeline_options) as p:
        (p
         | 'Read from GCS' >> beam.io.ReadFromText('gs://bucket/raw/jobstreet/*/dump.jsonl.gz')
         | 'Parse JSON' >> beam.Map(json.loads)
         | 'Validate Schema' >> beam.Map(validate_raw_job)
         | 'Clean & Transform' >> beam.Map(clean_job)
         | 'Write to BigQuery' >> beam.io.WriteToBigQuery(
             table='sg-job-market:sg_job_market.cleaned_jobs',
             schema=get_bq_schema(),
             write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
         ))