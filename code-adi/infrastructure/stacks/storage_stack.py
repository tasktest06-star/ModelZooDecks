"""StorageStack — S3 buckets and IAM roles for ADI Model Zoo MLOps."""
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_iam as iam,
    RemovalPolicy,
    Duration,
)
from constructs import Construct


class StorageStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Log bucket — retains access logs from other buckets
        self.log_bucket = s3.Bucket(
            self, "LogBucket",
            bucket_name="adi-modelzoo-logs",
            removal_policy=RemovalPolicy.RETAIN,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="ExpireLogsAfter90Days",
                    expiration=Duration.days(90),
                )
            ],
        )

        # Weights bucket — stores AI8X .pth.tar QAT checkpoints
        self.weights_bucket = s3.Bucket(
            self, "WeightsBucket",
            bucket_name="adi-modelzoo-weights",
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            server_access_logs_bucket=self.log_bucket,
            server_access_logs_prefix="weights/",
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="TransitionOldVersionsToIA",
                    noncurrent_version_transitions=[
                        s3.NoncurrentVersionTransition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(30),
                        )
                    ],
                    noncurrent_version_expiration=Duration.days(180),
                )
            ],
        )

        # Synthesized bucket — stores ai8xize-generated C headers and binaries
        self.synthesized_bucket = s3.Bucket(
            self, "SynthesizedBucket",
            bucket_name="adi-modelzoo-synthesized",
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            server_access_logs_bucket=self.log_bucket,
            server_access_logs_prefix="synthesized/",
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="TransitionSynthesizedToIA",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(30),
                        )
                    ],
                )
            ],
        )

        # Artifacts bucket — deployment bundles ready for MAX32690 / ADSP-SC835
        self.artifact_bucket = s3.Bucket(
            self, "ArtifactBucket",
            bucket_name="adi-modelzoo-artifacts",
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            server_access_logs_bucket=self.log_bucket,
            server_access_logs_prefix="artifacts/",
        )

        # IAM role for CodeBuild / CodePipeline
        self.pipeline_role = iam.Role(
            self, "PipelineRole",
            role_name="adi-modelzoo-pipeline-role",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("codebuild.amazonaws.com"),
                iam.ServicePrincipal("codepipeline.amazonaws.com"),
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryPowerUser"),
            ],
            inline_policies={
                "S3BucketAccess": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            actions=["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
                            resources=[
                                f"arn:aws:s3:::adi-modelzoo-weights",
                                f"arn:aws:s3:::adi-modelzoo-weights/*",
                                f"arn:aws:s3:::adi-modelzoo-synthesized",
                                f"arn:aws:s3:::adi-modelzoo-synthesized/*",
                                f"arn:aws:s3:::adi-modelzoo-artifacts",
                                f"arn:aws:s3:::adi-modelzoo-artifacts/*",
                                f"arn:aws:s3:::adi-modelzoo-logs",
                                f"arn:aws:s3:::adi-modelzoo-logs/*",
                                f"arn:aws:s3:::adi-modelzoo-pipeline-artifacts",
                                f"arn:aws:s3:::adi-modelzoo-pipeline-artifacts/*",
                            ],
                        )
                    ]
                )
            },
        )

        # Outputs
        cdk.CfnOutput(self, "WeightsBucketName", value=self.weights_bucket.bucket_name)
        cdk.CfnOutput(self, "SynthesizedBucketName", value=self.synthesized_bucket.bucket_name)
        cdk.CfnOutput(self, "ArtifactBucketName", value=self.artifact_bucket.bucket_name)
        cdk.CfnOutput(self, "PipelineRoleArn", value=self.pipeline_role.role_arn)
