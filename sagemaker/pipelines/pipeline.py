import os
import time 

import boto3
import sagemaker
import sagemaker.session

from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
    FrameworkProcessor,
)
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
    Join
)
from sagemaker.workflow.parameters import (
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.image_uris import retrieve


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
    """Gets the sagemaker client.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn.lower())
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name=None,
    dataset_prefix="dev_datasets",
    output_prefix="preprocessed_data",
    pipeline_name="Deepfake-detection-pipeline",
    base_job_prefix="Deepfake",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
):
    """Gets a SageMaker ML Pipeline instance working with on deepfake data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

    # parameters for pipeline execution
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    timestamp_prefix = str(int(time.time()))

    # processing step 
    pytorch_processor = FrameworkProcessor(
        estimator_cls=PyTorch,
        framework_version='1.12.0',
        py_version='py38',
        instance_type=processing_instance_type,
        instance_count=1, 
        base_job_name=f'{base_job_prefix}/deepfake-processing',
        sagemaker_session=pipeline_session,
        role=role
    )
    step_args = pytorch_processor.run(
        code="preprocess_deepfake.py",
        source_dir=os.path.join(BASE_DIR, 'preprocess'),
        outputs=[
            ProcessingOutput(
                output_name="train", 
                source="/opt/ml/processing/output/train",
                destination=Join(
                    on="/",
                    values=[
                        "s3://{}".format(default_bucket),
                        dataset_prefix,
                        output_prefix,
                        timestamp_prefix,
                        "train",
                    ],
                ),
            ),
            ProcessingOutput(
                output_name="validation", 
                source="/opt/ml/processing/output/validation",
                destination=Join(
                    on="/",
                    values=[
                        "s3://{}".format(default_bucket),
                        dataset_prefix,
                        output_prefix,
                        timestamp_prefix,
                        "validation",
                    ],
                ),
            ),
            ProcessingOutput(
                output_name="test", 
                source="/opt/ml/processing/output/test",
                destination=Join(
                    on="/",
                    values=[
                        "s3://{}".format(default_bucket),
                        dataset_prefix,
                        output_prefix,
                        timestamp_prefix,
                        "test",
                    ],
                ),
            ),
        ],
        arguments=['--frames_per_video', str(15),
                   '--batch_size', str(32),
                   '--face_size', str(224)
                  ]
    )
    step_preprocess_data = ProcessingStep(
        name="preprocess-deepfake-data",
        step_args=step_args
    )

    # training step for generating model artifacts
    # Define the output path for the model artifacts from the Training Job
    model_path = f"s3://{default_bucket}/{dataset_prefix}/training"
    est = PyTorch(
        entry_point="train.py",
        source_dir=os.path.join(BASE_DIR, "code"),  
        role=role,
        framework_version="1.12.1",
        py_version="py38",
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        hyperparameters={"batch-size": 32, 
                         "epochs": 5, 
                         "learning-rate": 1e-4
                        },
    )
    step_train_model = TrainingStep(
        name="train-deepfake-detection-model",
        estimator=est,
        inputs={
            "train": TrainingInput(
                s3_data=step_preprocess_data.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri
            ),
            "validation": TrainingInput(
                s3_data=step_preprocess_data.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri
            )
        }
    )

    # processing step for evaluation
    image_uri = retrieve(
        framework='pytorch', 
        region='us-east-1', 
        version='1.12.1', 
        py_version='py38', 
        image_scope='training',
        instance_type='ml.m5.xlarge'
    )
    evaluate_model_processor = ScriptProcessor(
        role=role,
        image_uri=image_uri,
        command=['python3'],
        instance_count=1,
        instance_type='ml.m5.xlarge'
    )
    evaluation_report = PropertyFile(
        name="AbaloneEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_evaluate_model = ProcessingStep(
        name="evaluate-deepfake-detection-model",
        processor=evaluate_model_processor,
        inputs=[
            ProcessingInput(
                source=step_train_model.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_preprocess_data.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation", 
                source="/opt/ml/processing/evaluation",
                destination = Join(
                    on="/",
                    values=[
                        "s3://{}".format(default_bucket),
                        dataset_prefix,
                        "evaluation",
                        timestamp_prefix,
                        "evaluation-report",
                    ],
                ),
            ),
        ],
        code=os.path.join(BASE_DIR, "evaluate/evaluate.py"),
        property_files=[evaluation_report]
    )

    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    step_evaluate_model.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"][
                        "S3Uri"
                    ],
                    "evaluation.json",
                ],
            ),
            content_type="application/json",
        )
    )
    step_register_model = RegisterModel(
        name="deepfake-detection",
        estimator=est,
        model_data=step_train_model.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/x-recordio-protobuf"], 
        response_types=["application/json"],
        inference_instances=["ml.m5.xlarge", "ml.m5.large"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    # condition step for evaluating model quality and branching execution
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_evaluate_model.name,
            property_file=evaluation_report,
            json_path="deepfake_detection_metrics.accuracy.value",
        ),
        right=0.40, #minimum accuracy value
    )
    step_cond = ConditionStep(
        name="check-accuracy-deepfake-evaluation",
        conditions=[cond_gte],
        if_steps=[step_register_model],
        else_steps=[],
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            training_instance_type,
            model_approval_status,
        ],
        steps=[step_preprocess_data, step_train_model, step_evaluate_model, step_cond],
        sagemaker_session=pipeline_session,
    )
    return pipeline
