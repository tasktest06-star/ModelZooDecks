"""Unit tests for ADI PipelineStack."""
import pytest
import aws_cdk as cdk
from aws_cdk import aws_s3 as s3, RemovalPolicy
from aws_cdk.assertions import Template, Match
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stacks.pipeline_stack import PipelineStack


@pytest.fixture(scope="module")
def template():
    app = cdk.App()
    env = cdk.Environment(account="123456789012", region="us-east-1")
    support = cdk.Stack(app, "SupportStack", env=env)
    weights_bucket = s3.Bucket(support, "WeightsBucket", removal_policy=RemovalPolicy.DESTROY)
    pipeline = PipelineStack(app, "TestAdiPipelineStack",
                             weights_bucket=weights_bucket,
                             env=env)
    return Template.from_stack(pipeline)


def test_qat_train_project_exists(template):
    template.has_resource_properties("AWS::CodeBuild::Project", {
        "Name": "adi-modelzoo-qat-train",
    })


def test_synthesis_project_exists(template):
    template.has_resource_properties("AWS::CodeBuild::Project", {
        "Name": "adi-modelzoo-synthesize",
    })


def test_eval_project_exists(template):
    template.has_resource_properties("AWS::CodeBuild::Project", {
        "Name": "adi-modelzoo-evaluate",
    })


def test_package_project_exists(template):
    template.has_resource_properties("AWS::CodeBuild::Project", {
        "Name": "adi-modelzoo-package",
    })


def test_pipeline_exists(template):
    template.has_resource_properties("AWS::CodePipeline::Pipeline", {
        "Name": "adi-modelzoo-mlops",
    })


def test_pipeline_has_four_stages(template):
    pipelines = template.find_resources("AWS::CodePipeline::Pipeline")
    for _, pipeline in pipelines.items():
        stages = pipeline.get("Properties", {}).get("Stages", [])
        assert len(stages) == 4, f"Expected 4 stages, got {len(stages)}"


def test_pipeline_artifact_bucket_exists(template):
    template.has_resource_properties("AWS::S3::Bucket", {
        "BucketName": "adi-modelzoo-pipeline-artifacts",
    })
