"""PipelineStack — 3-stage CodePipeline for TI EdgeAI: Source → Evaluate → Deploy."""
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_codebuild as codebuild,
    aws_codepipeline as codepipeline,
    aws_codepipeline_actions as actions,
    aws_s3 as s3,
    aws_iam as iam,
    Duration,
    RemovalPolicy,
)
from constructs import Construct
from stacks.storage_stack import StorageStack
from stacks.compute_stack import ComputeStack


class PipelineStack(Stack):
    def __init__(self, scope: Construct, construct_id: str,
                 storage_stack: StorageStack,
                 compute_stack: ComputeStack,
                 **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Own IAM role — literal S3 ARNs avoid cross-stack circular dependency
        build_role = iam.Role(
            self, "BuildRole",
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
                                "arn:aws:s3:::ti-edgeai-pipeline-artifacts",
                                "arn:aws:s3:::ti-edgeai-pipeline-artifacts/*",
                            ],
                        )
                    ]
                )
            },
        )

        shared_env = codebuild.BuildEnvironment(
            build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
            compute_type=codebuild.ComputeType.LARGE,
            privileged=True,
        )

        self.eval_project = codebuild.Project(
            self, "EvalProject",
            project_name="ti-edgeai-evaluate",
            environment=shared_env,
            role=build_role,
            build_spec=codebuild.BuildSpec.from_object({
                "version": "0.2",
                "phases": {
                    "install": {
                        "commands": ["pip install -r code/requirements.txt"]
                    },
                    "build": {
                        "commands": [
                            "aws s3 sync s3://ti-edgeai-models/ code/models/",
                            "python code/pipelines/eval_pipeline.py --config code/config/pipeline_config.yaml",
                        ]
                    },
                    "post_build": {
                        "commands": ["echo Evaluation complete"]
                    },
                },
                "reports": {
                    "eval-report": {"files": ["code/eval_results.json"]}
                },
            }),
            timeout=Duration.hours(2),
        )

        self.deploy_project = codebuild.Project(
            self, "DeployProject",
            project_name="ti-edgeai-deploy",
            environment=shared_env,
            role=build_role,
            build_spec=codebuild.BuildSpec.from_object({
                "version": "0.2",
                "phases": {
                    "build": {
                        "commands": [
                            "python code/pipelines/deploy_pipeline.py --config code/config/pipeline_config.yaml",
                            "aws s3 sync code/bundles/ s3://ti-edgeai-artifacts/",
                        ]
                    },
                },
            }),
            timeout=Duration.minutes(30),
        )

        pipeline_artifact_bucket = s3.Bucket(
            self, "PipelineArtifacts",
            bucket_name="ti-edgeai-pipeline-artifacts",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        source_output = codepipeline.Artifact("SourceOutput")
        eval_output = codepipeline.Artifact("EvalOutput")

        self.pipeline = codepipeline.Pipeline(
            self, "Pipeline",
            pipeline_name="ti-edgeai-mlops",
            artifact_bucket=pipeline_artifact_bucket,
            stages=[
                codepipeline.StageProps(
                    stage_name="Source",
                    actions=[
                        actions.S3SourceAction(
                            action_name="S3Source",
                            bucket=storage_stack.model_bucket,
                            bucket_key="trigger/trigger.zip",
                            output=source_output,
                        )
                    ],
                ),
                codepipeline.StageProps(
                    stage_name="Evaluate",
                    actions=[
                        actions.CodeBuildAction(
                            action_name="EvaluateModels",
                            project=self.eval_project,
                            input=source_output,
                            outputs=[eval_output],
                        )
                    ],
                ),
                codepipeline.StageProps(
                    stage_name="Deploy",
                    actions=[
                        actions.CodeBuildAction(
                            action_name="PackageArtifacts",
                            project=self.deploy_project,
                            input=eval_output,
                        )
                    ],
                ),
            ],
        )

        cdk.CfnOutput(self, "PipelineName", value=self.pipeline.pipeline_name)
        cdk.CfnOutput(self, "EvalProjectName", value=self.eval_project.project_name)
        cdk.CfnOutput(self, "DeployProjectName", value=self.deploy_project.project_name)
