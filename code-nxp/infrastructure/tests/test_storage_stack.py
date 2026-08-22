import pytest
import aws_cdk as cdk
from aws_cdk.assertions import Template, Match
from stacks.storage_stack import StorageStack


@pytest.fixture
def template():
    app = cdk.App()
    stack = StorageStack(app, "T", env=cdk.Environment(account="123456789012", region="us-east-1"))
    return Template.from_stack(stack)


class TestNXPStorageStack:
    def test_four_buckets_created(self, template):
        template.resource_count_is("AWS::S3::Bucket", 4)

    def test_tflite_bucket_versioned(self, template):
        template.has_resource_properties("AWS::S3::Bucket", {
            "VersioningConfiguration": {"Status": "Enabled"},
            "BucketName": Match.string_like_regexp("nxp-modelzoo-tflite.*"),
        })

    def test_vela_bucket_versioned(self, template):
        template.has_resource_properties("AWS::S3::Bucket", {
            "VersioningConfiguration": {"Status": "Enabled"},
            "BucketName": Match.string_like_regexp("nxp-modelzoo-vela.*"),
        })

    def test_all_buckets_encrypted(self, template):
        buckets = template.find_resources("AWS::S3::Bucket")
        for b in buckets.values():
            assert "BucketEncryption" in b.get("Properties", {}), "Bucket missing encryption"

    def test_all_buckets_block_public(self, template):
        buckets = template.find_resources("AWS::S3::Bucket")
        for b in buckets.values():
            pac = b.get("Properties", {}).get("PublicAccessBlockConfiguration", {})
            assert pac.get("BlockPublicAcls") is True

    def test_log_bucket_lifecycle(self, template):
        template.has_resource_properties("AWS::S3::Bucket", {
            "LifecycleConfiguration": {
                "Rules": Match.array_with([
                    Match.object_like({"ExpirationInDays": 90})
                ])
            }
        })

    def test_iam_role_for_codebuild(self, template):
        template.has_resource_properties("AWS::IAM::Role", {
            "AssumeRolePolicyDocument": {
                "Statement": Match.array_with([
                    Match.object_like({"Principal": {"Service": "codebuild.amazonaws.com"}})
                ])
            }
        })

    def test_outputs(self, template):
        template.has_output("TFLiteBucketName", {})
        template.has_output("VelaBucketName", {})
        template.has_output("ArtifactBucketName", {})
