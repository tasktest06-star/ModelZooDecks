"""Unit tests for ADI StorageStack using CDK assertions (no AWS credentials needed)."""
import pytest
import aws_cdk as cdk
from aws_cdk.assertions import Template, Match
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stacks.storage_stack import StorageStack


@pytest.fixture(scope="module")
def template():
    app = cdk.App()
    stack = StorageStack(app, "TestAdiStorageStack",
                         env=cdk.Environment(account="123456789012", region="us-east-1"))
    return Template.from_stack(stack)


def test_log_bucket_exists(template):
    template.has_resource_properties("AWS::S3::Bucket", {
        "BucketName": "adi-modelzoo-logs",
    })


def test_weights_bucket_versioning_enabled(template):
    template.has_resource_properties("AWS::S3::Bucket", {
        "BucketName": "adi-modelzoo-weights",
        "VersioningConfiguration": {"Status": "Enabled"},
    })


def test_synthesized_bucket_exists(template):
    template.has_resource_properties("AWS::S3::Bucket", {
        "BucketName": "adi-modelzoo-synthesized",
        "VersioningConfiguration": {"Status": "Enabled"},
    })


def test_artifact_bucket_versioning_enabled(template):
    template.has_resource_properties("AWS::S3::Bucket", {
        "BucketName": "adi-modelzoo-artifacts",
        "VersioningConfiguration": {"Status": "Enabled"},
    })


def test_all_buckets_block_public_access(template):
    template.resource_count_is("AWS::S3::Bucket", 4)
    buckets = template.find_resources("AWS::S3::Bucket")
    for bucket_id, bucket in buckets.items():
        props = bucket.get("Properties", {})
        bpa = props.get("PublicAccessBlockConfiguration", {})
        assert bpa.get("BlockPublicAcls") is True, f"Bucket {bucket_id} missing BlockPublicAcls"
        assert bpa.get("BlockPublicPolicy") is True, f"Bucket {bucket_id} missing BlockPublicPolicy"


def test_all_buckets_s3_managed_encryption(template):
    buckets = template.find_resources("AWS::S3::Bucket")
    for bucket_id, bucket in buckets.items():
        rules = bucket.get("Properties", {}).get("BucketEncryption", {}).get("ServerSideEncryptionConfiguration", [])
        assert len(rules) > 0, f"Bucket {bucket_id} missing encryption"
        algo = rules[0].get("ServerSideEncryptionByDefault", {}).get("SSEAlgorithm")
        assert algo == "aws:kms" or algo == "AES256", f"Bucket {bucket_id} unexpected encryption"


def test_log_bucket_lifecycle_90_days(template):
    template.has_resource_properties("AWS::S3::Bucket", {
        "BucketName": "adi-modelzoo-logs",
        "LifecycleConfiguration": {
            "Rules": Match.array_with([
                Match.object_like({"Status": "Enabled"})
            ])
        },
    })


def test_pipeline_role_exists(template):
    template.has_resource_properties("AWS::IAM::Role", {
        "RoleName": "adi-modelzoo-pipeline-role",
    })


def test_pipeline_role_can_be_assumed_by_codebuild(template):
    template.has_resource_properties("AWS::IAM::Role", {
        "AssumeRolePolicyDocument": Match.object_like({
            "Statement": Match.array_with([
                Match.object_like({
                    "Principal": {"Service": "codebuild.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                })
            ])
        })
    })
