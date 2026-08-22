"""Unit tests for ADI ComputeStack."""
import pytest
import aws_cdk as cdk
from aws_cdk.assertions import Template, Match
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stacks.compute_stack import ComputeStack


@pytest.fixture(scope="module")
def template():
    app = cdk.App()
    stack = ComputeStack(app, "TestAdiComputeStack",
                         env=cdk.Environment(account="123456789012", region="us-east-1"))
    return Template.from_stack(stack)


def test_training_ecr_repo_exists(template):
    template.has_resource_properties("AWS::ECR::Repository", {
        "RepositoryName": "adi-ai8x-training",
    })


def test_synthesis_ecr_repo_exists(template):
    template.has_resource_properties("AWS::ECR::Repository", {
        "RepositoryName": "adi-ai8x-synthesis",
    })


def test_ecr_repos_have_lifecycle_policy(template):
    repos = template.find_resources("AWS::ECR::Repository")
    for repo_id, repo in repos.items():
        lifecycle = repo.get("Properties", {}).get("LifecyclePolicy")
        assert lifecycle is not None, f"ECR repo {repo_id} missing LifecyclePolicy"


def test_ecs_cluster_exists(template):
    template.has_resource_properties("AWS::ECS::Cluster", {
        "ClusterName": "adi-modelzoo-eval-cluster",
        "ClusterSettings": Match.array_with([
            Match.object_like({"Name": "containerInsights", "Value": "enabled"})
        ]),
    })


def test_fargate_task_definition_exists(template):
    template.has_resource_properties("AWS::ECS::TaskDefinition", {
        "Family": "adi-modelzoo-eval",
        "RequiresCompatibilities": ["FARGATE"],
        "Cpu": "2048",
        "Memory": "4096",
    })


def test_spot_instance_role_exists(template):
    template.has_resource_properties("AWS::IAM::Role", {
        "RoleName": "adi-modelzoo-spot-instance-role",
        "AssumeRolePolicyDocument": Match.object_like({
            "Statement": Match.array_with([
                Match.object_like({
                    "Principal": {"Service": "ec2.amazonaws.com"},
                })
            ])
        })
    })


def test_instance_profile_exists(template):
    template.has_resource_properties("AWS::IAM::InstanceProfile", {
        "InstanceProfileName": "adi-modelzoo-spot-profile",
    })


def test_launch_template_g4dn_xlarge(template):
    template.has_resource_properties("AWS::EC2::LaunchTemplate", {
        "LaunchTemplateData": Match.object_like({
            "InstanceType": "g4dn.xlarge",
        })
    })
