"""Compute for NXP eIQ Model Zoo: ECR + ECS Fargate for Docker recipe.sh execution."""

import aws_cdk as cdk
from aws_cdk import (
    Stack, aws_ecr as ecr, aws_ec2 as ec2, aws_iam as iam,
    aws_ecs as ecs, RemovalPolicy,
)
from constructs import Construct


class ComputeStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        self.vpc = ec2.Vpc(
            self, "NXPVpc",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="Private", subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS, cidr_mask=24
                ),
            ],
        )

        # ECR: nxp-model-zoo Docker image (runs recipe.sh)
        self.recipe_repo = ecr.Repository(
            self, "RecipeRepo",
            repository_name="nxp-eiq-recipe-runner",
            removal_policy=RemovalPolicy.DESTROY,
            lifecycle_rules=[ecr.LifecycleRule(max_image_count=5)],
        )

        # ECR: Vela compiler image
        self.vela_repo = ecr.Repository(
            self, "VelaRepo",
            repository_name="nxp-vela-compiler",
            removal_policy=RemovalPolicy.DESTROY,
            lifecycle_rules=[ecr.LifecycleRule(max_image_count=5)],
        )

        self.cluster = ecs.Cluster(
            self, "NXPCluster",
            cluster_name="nxp-modelzoo-cluster",
            vpc=self.vpc,
            container_insights=True,
        )

        # IAM role for Fargate tasks
        self.task_role = iam.Role(
            self, "FargateTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3ReadOnlyAccess"),
            ],
        )

        self.task_execution_role = iam.Role(
            self, "FargateExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonECSTaskExecutionRolePolicy"
                ),
            ],
        )

        # Fargate task definition for recipe.sh execution
        self.recipe_task_def = ecs.FargateTaskDefinition(
            self, "RecipeTaskDef",
            family="nxp-recipe-runner",
            cpu=4096,
            memory_limit_mib=8192,
            task_role=self.task_role,
            execution_role=self.task_execution_role,
        )
        self.recipe_task_def.add_container(
            "RecipeContainer",
            image=ecs.ContainerImage.from_ecr_repository(self.recipe_repo, tag="latest"),
            logging=ecs.LogDrivers.aws_logs(stream_prefix="nxp-recipe"),
            environment={
                "MODEL_ZOO_ROOT": "/workspace",
                "OUTPUT_DIR": "/workspace/output",
            },
        )

        cdk.CfnOutput(self, "RecipeRepoUri", value=self.recipe_repo.repository_uri)
        cdk.CfnOutput(self, "VelaRepoUri", value=self.vela_repo.repository_uri)
        cdk.CfnOutput(self, "ClusterName", value=self.cluster.cluster_name)
