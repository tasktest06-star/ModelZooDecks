import pytest
import aws_cdk as cdk
from aws_cdk.assertions import Template, Match
from stacks.compute_stack import ComputeStack


@pytest.fixture
def template():
    app = cdk.App()
    stack = ComputeStack(app, "T", env=cdk.Environment(account="123456789012", region="us-east-1"))
    return Template.from_stack(stack)


class TestNXPComputeStack:
    def test_two_ecr_repos(self, template):
        template.resource_count_is("AWS::ECR::Repository", 2)

    def test_recipe_repo_name(self, template):
        template.has_resource_properties("AWS::ECR::Repository", {
            "RepositoryName": "nxp-eiq-recipe-runner",
        })

    def test_vela_repo_name(self, template):
        template.has_resource_properties("AWS::ECR::Repository", {
            "RepositoryName": "nxp-vela-compiler",
        })

    def test_ecs_cluster_created(self, template):
        template.resource_count_is("AWS::ECS::Cluster", 1)

    def test_cluster_container_insights(self, template):
        template.has_resource_properties("AWS::ECS::Cluster", {
            "ClusterSettings": Match.array_with([
                Match.object_like({"Name": "containerInsights", "Value": "enabled"})
            ])
        })

    def test_fargate_task_definition(self, template):
        template.has_resource_properties("AWS::ECS::TaskDefinition", {
            "Family": "nxp-recipe-runner",
            "Cpu": "4096",
            "Memory": "8192",
        })

    def test_vpc_created(self, template):
        template.resource_count_is("AWS::EC2::VPC", 1)

    def test_outputs(self, template):
        template.has_output("RecipeRepoUri", {})
        template.has_output("ClusterName", {})
