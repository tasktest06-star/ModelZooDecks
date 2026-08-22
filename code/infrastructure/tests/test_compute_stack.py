"""Unit tests for TI ComputeStack."""
import pytest
import aws_cdk as cdk
from aws_cdk.assertions import Template, Match
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stacks.compute_stack import ComputeStack


@pytest.fixture(scope="module")
def template():
    app = cdk.App()
    stack = ComputeStack(app, "TestTiComputeStack",
                         env=cdk.Environment(account="123456789012", region="us-east-1"))
    return Template.from_stack(stack)


def test_ecr_repo_exists(template):
    template.has_resource_properties("AWS::ECR::Repository", {"RepositoryName": "ti-edgeai-eval"})


def test_ecr_repo_lifecycle_policy(template):
    repos = template.find_resources("AWS::ECR::Repository")
    for repo_id, repo in repos.items():
        assert repo.get("Properties", {}).get("LifecyclePolicy") is not None


def test_ecs_cluster_exists(template):
    template.has_resource_properties("AWS::ECS::Cluster", {
        "ClusterName": "ti-edgeai-eval-cluster",
        "ClusterSettings": Match.array_with([
            Match.object_like({"Name": "containerInsights", "Value": "enabled"})
        ]),
    })


def test_fargate_task_definition(template):
    template.has_resource_properties("AWS::ECS::TaskDefinition", {
        "Family": "ti-edgeai-eval",
        "RequiresCompatibilities": ["FARGATE"],
        "Cpu": "2048",
        "Memory": "4096",
    })


def test_vpc_created(template):
    template.resource_count_is("AWS::EC2::VPC", 1)
