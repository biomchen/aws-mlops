
import sys

from pyspark.context import SparkContext

from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions

from constants import ARGS

class JobCreator:

    def __init__(self):
        self.args = getResolvedOptions(sys.argv, ARGS)
        self.glueContext, self.spark = self.initialize_glue()

    def initialize_glue(self):
        sc = SparkContext()
        glueContext = GlueContext(sc)
        spark = glueContext.spark_session
        return glueContext, spark
    
    def initialize_job(self):
        job = Job(self.glueContext)
        job.init(self.args['JOB_NAME'], self.args)
        return job
    
    def load_source_data(self):
        return self.spark.read.load(
            self.args['S3_SOURCE'],
            format='csv',
            infraSchema=True,
            header=False
        )
    
    def process_job(self):
        job = self.initialize_job()
        self.create_train_test_data()
        job.commit()

    def create_train_test_data(self):
       data = self.load_source_data()
       train_data, test_data = data.randomSplit([0.7, 0.3])
       train_path = self.args['S3_DEST'] + self.args['TRAIN_KEY']
       test_path = self.args['S3_DEST'] + self.args['VAL_KEY']
       train_data.write.save(train_path, format='csv', model='overwrite')
       test_data.write.save(test_path, format='csv', model='overwrite')

JobCreator().process_job()
