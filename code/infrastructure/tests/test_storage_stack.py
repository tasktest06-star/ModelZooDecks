"""Unit tests for TI StorageStack."""
import pytest
import aws_cdk as cdk
from aws_cdk.assertions import Template, Match
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stacks.storage_stack import StorageStack


@pytest.fixture(scope="module")
def template():
    app = cdk.App()
    stack = StorageStack(app, "TestTiStorageStack",
                         env=cdk.Environment(account="123456789012", region="us-east-1"))
    return Template.from_stack(stack)


def test_three_buckets_created(template):
    template.resource_count_is("AWS::S3::Bucket", 3)


def test_log_bucket_exists(template):
    template.has_resource_properties("AWS::S3::Bucket", {"BucketName": "ti-edgeai-modelzoo-logs"})


def test_model_bucket_versioned(template):
    template.has_resource_properties("AWS::S3::Bucket", {
        "BucketName": "ti-edgeai-models",
        "VersioningConfiguration": {"Status": "Enabled"},
    })


def test_artifact_bucket_versioned(template):
    template.has_resource_properties("AWS::S3::Bucket", {
        "BucketName": "ti-edgeai-artifacts",
        "VersioningConfiguration": {"Status": "Enabled"},
    })


def test_all_buckets_block_public_access(template):
    buckets = template.find_resources("AWS::S3::Bucket")
    for bucket_id, bucket in buckets.items():
        bpa = bucket.get("Properties", {}).get("PublicAccessBlockConfiguration", {})
        assert bpa.get("BlockPublicAcls") is True
        assert bpa.get("BlockPublicPolicy") is True


def test_all_buckets_encrypted(template):
    buckets = template.find_resources("AWS::S3::Bucket")
    for bucket_id, bucket in buckets.items():
        rules = bucket.get("Properties", {}).get("BucketEncryption", {}).get("ServerSideEncryptionConfiguration", [])
        assert len(rules) > 0, f"Bucket {bucket_id} missing encryption"


def test_pipeline_role_exists(template):
    template.has_resource_properties("AWS::IAM::Role", {"RoleName": "ti-edgeai-pipeline-role"})


def test_pipeline_role_codebuild_assumption(template):
    template.has_resource_properties("AWS::IAM::Role", {
        "AssumeRolePolicyDocument": Match.object_like({
            "Statement": Match.array_with([
                Match.object_like({"Principal": {"Service": "codebuild.amazonaws.com"}})
            ])
        })
    })
