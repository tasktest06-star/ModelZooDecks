import pytest
import aws_cdk as cdk
from aws_cdk.assertions import Template, Match
from stacks.storage_stack import StorageStack
from stacks.pipeline_stack import PipelineStack


@pytest.fixture
def template():
    app = cdk.App()
    env = cdk.Environment(account="123456789012", region="us-east-1")
    storage = StorageStack(app, "S", env=env)
    pipeline = PipelineStack(app, "P",
                              tflite_bucket=storage.tflite_bucket,
                              vela_bucket=storage.vela_bucket,
                              artifact_bucket=storage.artifact_bucket,
                              env=env)
    return Template.from_stack(pipeline)


class TestNXPPipelineStack:
    def test_pipeline_name(self, template):
        template.has_resource_properties("AWS::CodePipeline::Pipeline", {
            "Name": "nxp-modelzoo-mlops",
        })

    def test_four_stages(self, template):
        template.has_resource_properties("AWS::CodePipeline::Pipeline", {
            "Stages": Match.array_with([
                Match.object_like({"Name": "Source"}),
                Match.object_like({"Name": "BuildRecipes"}),
                Match.object_like({"Name": "CompileVela"}),
                Match.object_like({"Name": "Package"}),
            ])
        })

    def test_three_codebuild_projects(self, template):
        template.resource_count_is("AWS::CodeBuild::Project", 3)

    def test_recipe_project_name(self, template):
        template.has_resource_properties("AWS::CodeBuild::Project", {
            "Name": "nxp-build-recipes",
        })

    def test_vela_project_name(self, template):
        template.has_resource_properties("AWS::CodeBuild::Project", {
            "Name": "nxp-vela-compile",
        })

    def test_outputs(self, template):
        template.has_output("PipelineName", {})
        template.has_output("VelaProjectName", {})
