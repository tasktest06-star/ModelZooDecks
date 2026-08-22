"""S3 storage for NXP eIQ Model Zoo: TFLite INT8 models, Vela-compiled models, artifacts."""

import aws_cdk as cdk
from aws_cdk import Stack, aws_s3 as s3, aws_iam as iam, RemovalPolicy, Duration
from constructs import Construct


class StorageStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        self.log_bucket = s3.Bucket(
            self, "LogBucket",
            bucket_name=f"nxp-modelzoo-logs-{self.account}-{self.region}",
            removal_policy=RemovalPolicy.RETAIN,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            lifecycle_rules=[s3.LifecycleRule(expiration=Duration.days(90))],
        )

        # TFLite INT8 models built by recipe.sh
        self.tflite_bucket = s3.Bucket(
            self, "TFLiteBucket",
            bucket_name=f"nxp-modelzoo-tflite-{self.account}-{self.region}",
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            server_access_logs_bucket=self.log_bucket,
            server_access_logs_prefix="tflite/",
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="RetainRecentVersions",
                    noncurrent_version_expiration=Duration.days(60),
                )
            ],
        )

        # Vela-compiled *_vela.tflite for i.MX 93 + Ethos-U65
        self.vela_bucket = s3.Bucket(
            self, "VelaBucket",
            bucket_name=f"nxp-modelzoo-vela-{self.account}-{self.region}",
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            server_access_logs_bucket=self.log_bucket,
            server_access_logs_prefix="vela/",
        )

        # Deployment bundles: TFLite + Vela + recipe.sh + manifest.json + inference_example.py
        self.artifact_bucket = s3.Bucket(
            self, "ArtifactBucket",
            bucket_name=f"nxp-modelzoo-artifacts-{self.account}-{self.region}",
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            server_access_logs_bucket=self.log_bucket,
            server_access_logs_prefix="artifacts/",
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="TransitionToIA",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(30),
                        )
                    ],
                )
            ],
        )

        self.build_role = iam.Role(
            self, "BuildRole",
            assumed_by=iam.ServicePrincipal("codebuild.amazonaws.com"),
            description="CodeBuild access to NXP eIQ Model Zoo S3 buckets",
        )
        for bucket in [self.tflite_bucket, self.vela_bucket, self.artifact_bucket]:
            bucket.grant_read_write(self.build_role)

        cdk.CfnOutput(self, "TFLiteBucketName", value=self.tflite_bucket.bucket_name)
        cdk.CfnOutput(self, "VelaBucketName", value=self.vela_bucket.bucket_name)
        cdk.CfnOutput(self, "ArtifactBucketName", value=self.artifact_bucket.bucket_name)
