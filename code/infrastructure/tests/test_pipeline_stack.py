"""Unit tests for TI PipelineStack."""
import pytest
import aws_cdk as cdk
from aws_cdk.assertions import Template, Match
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stacks.storage_stack import StorageStack
from stacks.pipeline_stack import PipelineStack


@pytest.fixture(scope="module")
def template():
    app = cdk.App()
    env = cdk.Environment(account="123456789012", region="us-east-1")
    storage = StorageStack(app, "TestTiStorageForPipeline", env=env)
    pipeline = PipelineStack(app, "TestTiPipelineStack",
                             storage_stack=storage,
                             env=env)
    return Template.from_stack(pipeline)


def test_eval_project_exists(template):
    template.has_resource_properties("AWS::CodeBuild::Project", {"Name": "ti-edgeai-evaluate"})


def test_deploy_project_exists(template):
    template.has_resource_properties("AWS::CodeBuild::Project", {"Name": "ti-edgeai-deploy"})


def test_pipeline_exists(template):
    template.has_resource_properties("AWS::CodePipeline::Pipeline", {"Name": "ti-edgeai-mlops"})


def test_pipeline_has_three_stages(template):
    pipelines = template.find_resources("AWS::CodePipeline::Pipeline")
    for _, pipeline in pipelines.items():
        stages = pipeline.get("Properties", {}).get("Stages", [])
        assert len(stages) == 3, f"Expected 3 stages, got {len(stages)}"


def test_pipeline_artifact_bucket_exists(template):
    template.has_resource_properties("AWS::S3::Bucket", {"BucketName": "ti-edgeai-pipeline-artifacts"})
