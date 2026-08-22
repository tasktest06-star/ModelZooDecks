"""PipelineStack — 4-stage CodePipeline for ADI QAT train → synthesize → package."""
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_codebuild as codebuild,
    aws_codepipeline as codepipeline,
    aws_codepipeline_actions as actions,
    aws_s3 as s3,
    aws_iam as iam,
    Duration,
)
from constructs import Construct


class PipelineStack(Stack):
    def __init__(self, scope: Construct, construct_id: str,
                 weights_bucket: s3.IBucket,
                 **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Own IAM role — avoids circular cross-stack dependency
        build_role = iam.Role(
            self, "BuildRole",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("codebuild.amazonaws.com"),
                iam.ServicePrincipal("codepipeline.amazonaws.com"),
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonEC2ContainerRegistryPowerUser"
                ),
            ],
            inline_policies={
                "S3BucketAccess": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            actions=["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
                            resources=[
                                "arn:aws:s3:::adi-modelzoo-weights-*",
                                "arn:aws:s3:::adi-modelzoo-weights-*/*",
                                "arn:aws:s3:::adi-modelzoo-synthesized-*",
                                "arn:aws:s3:::adi-modelzoo-synthesized-*/*",
                                "arn:aws:s3:::adi-modelzoo-artifacts-*",
                                "arn:aws:s3:::adi-modelzoo-artifacts-*/*",
                                "arn:aws:s3:::adi-modelzoo-pipeline-artifacts*",
                                "arn:aws:s3:::adi-modelzoo-pipeline-artifacts*/*",
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

        # Stage 1 — QAT Training (kicks off spot EC2 via run-instances, waits for S3 checkpoint)
        self.qat_train_project = codebuild.Project(
            self, "QATTrainProject",
            project_name="adi-modelzoo-qat-train",
            environment=shared_env,
            role=build_role,
            build_spec=codebuild.BuildSpec.from_object({
                "version": "0.2",
                "phases": {
                    "install": {
                        "commands": [
                            "pip install boto3 pyyaml",
                        ]
                    },
                    "build": {
                        "commands": [
                            "echo Starting QAT training via spot instance",
                            "python code-adi/pipelines/train_pipeline.py --config code-adi/config/pipeline_config.yaml",
                            "aws s3 sync code-adi/checkpoints/ s3://adi-modelzoo-weights/checkpoints/",
                        ]
                    },
                    "post_build": {
                        "commands": [
                            "echo Training complete, weights uploaded",
                        ]
                    },
                },
                "artifacts": {
                    "files": ["code-adi/checkpoints/**/*"],
                },
            }),
            timeout=Duration.hours(4),
        )

        # Stage 2 — ai8xize Synthesis (generates C headers from trained weights)
        self.synthesis_project = codebuild.Project(
            self, "SynthesisProject",
            project_name="adi-modelzoo-synthesize",
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                compute_type=codebuild.ComputeType.LARGE,
                privileged=True,
            ),
            role=build_role,
            build_spec=codebuild.BuildSpec.from_object({
                "version": "0.2",
                "phases": {
                    "install": {
                        "commands": [
                            "aws s3 sync s3://adi-modelzoo-weights/checkpoints/ checkpoints/",
                        ]
                    },
                    "build": {
                        "commands": [
                            "python code-adi/pipelines/train_pipeline.py --config code-adi/config/pipeline_config.yaml --stage synthesize",
                            "aws s3 sync code-adi/synthesized/ s3://adi-modelzoo-synthesized/",
                        ]
                    },
                },
                "artifacts": {
                    "files": ["code-adi/synthesized/**/*"],
                },
            }),
            timeout=Duration.hours(2),
        )

        # Stage 3 — Evaluation
        self.eval_project = codebuild.Project(
            self, "EvalProject",
            project_name="adi-modelzoo-evaluate",
            environment=shared_env,
            role=build_role,
            build_spec=codebuild.BuildSpec.from_object({
                "version": "0.2",
                "phases": {
                    "install": {
                        "commands": [
                            "pip install -r code-adi/requirements.txt",
                        ]
                    },
                    "build": {
                        "commands": [
                            "aws s3 sync s3://adi-modelzoo-synthesized/ code-adi/synthesized/",
                            "python code-adi/pipelines/eval_pipeline.py --config code-adi/config/pipeline_config.yaml",
                        ]
                    },
                    "post_build": {
                        "commands": [
                            "echo Evaluation complete",
                        ]
                    },
                },
                "reports": {
                    "eval-report": {
                        "files": ["code-adi/eval_results.json"],
                    }
                },
            }),
            timeout=Duration.hours(1),
        )

        # Stage 4 — Package artifacts for MAX32690 / ADSP-SC835
        self.package_project = codebuild.Project(
            self, "PackageProject",
            project_name="adi-modelzoo-package",
            environment=shared_env,
            role=build_role,
            build_spec=codebuild.BuildSpec.from_object({
                "version": "0.2",
                "phases": {
                    "build": {
                        "commands": [
                            "python code-adi/pipelines/deploy_pipeline.py --config code-adi/config/pipeline_config.yaml",
                            "aws s3 sync code-adi/bundles/ s3://adi-modelzoo-artifacts/",
                        ]
                    },
                },
            }),
            timeout=Duration.minutes(30),
        )

        # Pipeline artifact bucket (CodePipeline internal)
        pipeline_artifact_bucket = s3.Bucket(
            self, "PipelineArtifacts",
            bucket_name="adi-modelzoo-pipeline-artifacts",
            removal_policy=cdk.RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        source_output = codepipeline.Artifact("SourceOutput")
        train_output = codepipeline.Artifact("TrainOutput")
        synth_output = codepipeline.Artifact("SynthOutput")
        eval_output = codepipeline.Artifact("EvalOutput")

        self.pipeline = codepipeline.Pipeline(
            self, "Pipeline",
            pipeline_name="adi-modelzoo-mlops",
            artifact_bucket=pipeline_artifact_bucket,
            stages=[
                codepipeline.StageProps(
                    stage_name="Source",
                    actions=[
                        actions.S3SourceAction(
                            action_name="S3Source",
                            bucket=weights_bucket,
                            bucket_key="trigger/trigger.zip",
                            output=source_output,
                        )
                    ],
                ),
                codepipeline.StageProps(
                    stage_name="QATTrain",
                    actions=[
                        actions.CodeBuildAction(
                            action_name="QATTrain",
                            project=self.qat_train_project,
                            input=source_output,
                            outputs=[train_output],
                        )
                    ],
                ),
                codepipeline.StageProps(
                    stage_name="Synthesize",
                    actions=[
                        actions.CodeBuildAction(
                            action_name="AI8XizeSynthesize",
                            project=self.synthesis_project,
                            input=train_output,
                            outputs=[synth_output],
                        )
                    ],
                ),
                codepipeline.StageProps(
                    stage_name="EvaluateAndPackage",
                    actions=[
                        actions.CodeBuildAction(
                            action_name="Evaluate",
                            project=self.eval_project,
                            input=synth_output,
                            outputs=[eval_output],
                            run_order=1,
                        ),
                        actions.CodeBuildAction(
                            action_name="Package",
                            project=self.package_project,
                            input=eval_output,
                            run_order=2,
                        ),
                    ],
                ),
            ],
        )

        cdk.CfnOutput(self, "PipelineName", value=self.pipeline.pipeline_name)
        cdk.CfnOutput(self, "QATTrainProjectName", value=self.qat_train_project.project_name)
        cdk.CfnOutput(self, "SynthesisProjectName", value=self.synthesis_project.project_name)
        cdk.CfnOutput(self, "EvalProjectName", value=self.eval_project.project_name)
