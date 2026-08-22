"""StorageStack — S3 buckets and IAM role for TI EdgeAI Model Zoo MLOps."""
import aws_cdk as cdk
from aws_cdk import Stack, aws_s3 as s3, aws_iam as iam, RemovalPolicy, Duration
from constructs import Construct


class StorageStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.log_bucket = s3.Bucket(
            self, "LogBucket",
            bucket_name="ti-edgeai-modelzoo-logs",
            removal_policy=RemovalPolicy.RETAIN,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            lifecycle_rules=[s3.LifecycleRule(expiration=Duration.days(90))],
        )

        # Model bucket — ONNX/TFLite/TIDL model files
        self.model_bucket = s3.Bucket(
            self, "ModelBucket",
            bucket_name="ti-edgeai-models",
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            server_access_logs_bucket=self.log_bucket,
            server_access_logs_prefix="models/",
            lifecycle_rules=[
                s3.LifecycleRule(
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

        # Artifact bucket — deployment bundles for TDA4VM/AM62A/AM68A/AM69A
        self.artifact_bucket = s3.Bucket(
            self, "ArtifactBucket",
            bucket_name="ti-edgeai-artifacts",
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            server_access_logs_bucket=self.log_bucket,
            server_access_logs_prefix="artifacts/",
            lifecycle_rules=[
                s3.LifecycleRule(
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(30),
                        )
                    ],
                )
            ],
        )

        # IAM role — uses literal ARNs to avoid cross-stack circular dependency
        self.pipeline_role = iam.Role(
            self, "PipelineRole",
            role_name="ti-edgeai-pipeline-role",
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
                                "arn:aws:s3:::ti-edgeai-models",
                                "arn:aws:s3:::ti-edgeai-models/*",
                                "arn:aws:s3:::ti-edgeai-artifacts",
                                "arn:aws:s3:::ti-edgeai-artifacts/*",
                                "arn:aws:s3:::ti-edgeai-modelzoo-logs",
                                "arn:aws:s3:::ti-edgeai-modelzoo-logs/*",
                                "arn:aws:s3:::ti-edgeai-pipeline-artifacts",
                                "arn:aws:s3:::ti-edgeai-pipeline-artifacts/*",
                            ],
                        )
                    ]
                )
            },
        )

        cdk.CfnOutput(self, "ModelBucketName", value=self.model_bucket.bucket_name)
        cdk.CfnOutput(self, "ArtifactBucketName", value=self.artifact_bucket.bucket_name)
        cdk.CfnOutput(self, "PipelineRoleArn", value=self.pipeline_role.role_arn)
