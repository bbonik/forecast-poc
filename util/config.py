"""Environment configuration/discovery utilities for Forecast PoC exercises"""

# External Dependencies:
import boto3


cfn = boto3.resource("cloudformation")


def find_solution_data_bucket() -> str:
    """Look up the name of the data bucket created by an Amazon Forecast MLOps solution deployment

    Given that (exactly one copy of) the 'Improving Forecast Accuracy with Machine Learning' AWS
    Solution has been deployed in this account and region, look up the name of the created Amazon
    S3 data bucket.

    This function requires your SageMaker execution role to have AWS CloudFormation read access.
    """
    try:
        bucket_output = next(
            o
            for stack in cfn.stacks.all() if (stack.parent_id is not None)
            for o in (stack.outputs or []) if (o["OutputKey"] == "ForecastBucketName")
        )
    except StopIteration as e:
        raise RuntimeError(
            "Couldn't find a nested CloudFormation stack with output 'ForecastBucketName'"
        ) from e
 
    return bucket_output["OutputValue"]
