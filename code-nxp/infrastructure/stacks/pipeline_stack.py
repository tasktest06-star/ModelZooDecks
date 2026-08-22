"""CodePipeline for NXP eIQ: BuildRecipes → CompileVela → Evaluate → Package."""

import aws_cdk as cdk
from aws_cdk import (
    Stack, aws_codebuild as codebuild, aws_codepipeline as codepipeline,
    aws_codepipeline_actions as actions, aws_s3 as s3, aws_iam as iam,
)
from constructs import Construct


class PipelineStack(Stack):
    def __init__(self, scope, construct_id, tflite_bucket, vela_bucket,
                 artifact_bucket, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        build_role = iam.Role(
            self, "BuildRole",
            assumed_by=iam.ServicePrincipal("codebuild.amazonaws.com"),
        )
        for bucket in [tflite_bucket, vela_bucket, artifact_bucket]:
            bucket.grant_read_write(build_role)

        # Stage 1: run recipe.sh via Docker to build TFLite INT8 models
        self.recipe_project = codebuild.Project(
            self, "RecipeProject",
            project_name="nxp-build-recipes",
            role=build_role,
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                compute_type=codebuild.ComputeType.LARGE,
                privileged=True,
                environment_variables={
                    "TFLITE_BUCKET": codebuild.BuildEnvironmentVariable(
                        value=tflite_bucket.bucket_name
                    ),
                },
            ),
            build_spec=codebuild.BuildSpec.from_object({
                "version": "0.2",
                "phases": {
                    "install": {
                        "runtime-versions": {"python": "3.10"},
                        "commands": ["pip install -r code-nxp/requirements.txt"],
                    },
                    "build": {
                        "commands": [
                            "cd code-nxp",
                            "python -m pytest tests/ -v --tb=short",
                            "python pipelines/eval_pipeline.py "
                            "--config config/model_registry.yaml --platform imx8mplus",
                            "aws s3 sync output/ s3://$TFLITE_BUCKET/models/",
                        ],
                    },
                },
            }),
        )

        # Stage 2: Vela compilation for i.MX 93 + Ethos-U65
        self.vela_project = codebuild.Project(
            self, "VelaProject",
            project_name="nxp-vela-compile",
            role=build_role,
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                compute_type=codebuild.ComputeType.MEDIUM,
                environment_variables={
                    "TFLITE_BUCKET": codebuild.BuildEnvironmentVariable(
                        value=tflite_bucket.bucket_name
                    ),
                    "VELA_BUCKET": codebuild.BuildEnvironmentVariable(
                        value=vela_bucket.bucket_name
                    ),
                },
            ),
            build_spec=codebuild.BuildSpec.from_object({
                "version": "0.2",
                "phases": {
                    "install": {
                        "commands": [
                            "pip install ethos-u-vela",
                            "pip install -r code-nxp/requirements.txt",
                        ]
                    },
                    "build": {
                        "commands": [
                            "aws s3 sync s3://$TFLITE_BUCKET/models/ tflite_models/",
                            "cd code-nxp",
                            "python mlops/vela_compiler.py "
                            "--input-dir ../tflite_models --accelerator ethos-u65-256",
                            "aws s3 sync vela_output/ s3://$VELA_BUCKET/",
                        ],
                    },
                },
            }),
        )

        # Stage 3: package deployment bundles
        self.package_project = codebuild.Project(
            self, "PackageProject",
            project_name="nxp-package-artifacts",
            role=build_role,
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                compute_type=codebuild.ComputeType.SMALL,
                environment_variables={
                    "ARTIFACT_BUCKET": codebuild.BuildEnvironmentVariable(
                        value=artifact_bucket.bucket_name
                    ),
                },
            ),
            build_spec=codebuild.BuildSpec.from_object({
                "version": "0.2",
                "phases": {
                    "install": {"commands": ["pip install -r code-nxp/requirements.txt"]},
                    "build": {
                        "commands": [
                            "cd code-nxp",
                            "python pipelines/deploy_pipeline.py "
                            "--config config/model_registry.yaml "
                            "--model mobilenetv2 --platform imx8mplus --target local",
                            "aws s3 sync deployed/ s3://$ARTIFACT_BUCKET/bundles/",
                        ],
                    },
                },
            }),
        )

        source_out = codepipeline.Artifact("Source")
        recipe_out = codepipeline.Artifact("RecipeOut")
        vela_out = codepipeline.Artifact("VelaOut")

        self.pipeline = codepipeline.Pipeline(
            self, "NXPPipeline",
            pipeline_name="nxp-modelzoo-mlops",
            artifact_bucket=artifact_bucket,
            stages=[
                codepipeline.StageProps(
                    stage_name="Source",
                    actions=[actions.S3SourceAction(
                        action_name="S3Source",
                        bucket=tflite_bucket,
                        bucket_key="source/modelzoo.zip",
                        output=source_out,
                    )],
                ),
                codepipeline.StageProps(
                    stage_name="BuildRecipes",
                    actions=[actions.CodeBuildAction(
                        action_name="RunRecipes",
                        project=self.recipe_project,
                        input=source_out,
                        outputs=[recipe_out],
                    )],
                ),
                codepipeline.StageProps(
                    stage_name="CompileVela",
                    actions=[actions.CodeBuildAction(
                        action_name="VelaCompile",
                        project=self.vela_project,
                        input=recipe_out,
                        outputs=[vela_out],
                    )],
                ),
                codepipeline.StageProps(
                    stage_name="Package",
                    actions=[actions.CodeBuildAction(
                        action_name="PackageArtifacts",
                        project=self.package_project,
                        input=vela_out,
                    )],
                ),
            ],
        )

        cdk.CfnOutput(self, "PipelineName", value=self.pipeline.pipeline_name)
        cdk.CfnOutput(self, "RecipeProjectName", value=self.recipe_project.project_name)
        cdk.CfnOutput(self, "VelaProjectName", value=self.vela_project.project_name)
