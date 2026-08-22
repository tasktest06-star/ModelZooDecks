"""ComputeStack — ECR and ECS Fargate resources for TI EdgeAI evaluation."""
import aws_cdk as cdk
from aws_cdk import Stack, aws_ecr as ecr, aws_ecs as ecs, aws_ec2 as ec2, RemovalPolicy
from constructs import Construct


class ComputeStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ECR — evaluation image (includes TIDL tools, onnxruntime, tflite)
        self.ecr_repo = ecr.Repository(
            self, "EvalRepo",
            repository_name="ti-edgeai-eval",
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                ecr.LifecycleRule(
                    description="Keep latest 10 eval images",
                    max_image_count=10,
                    rule_priority=1,
                )
            ],
        )

        self.vpc = ec2.Vpc(
            self, "Vpc",
            vpc_name="ti-edgeai-vpc",
            max_azs=2,
            nat_gateways=1,
        )

        self.cluster = ecs.Cluster(
            self, "EvalCluster",
            cluster_name="ti-edgeai-eval-cluster",
            vpc=self.vpc,
            container_insights=True,
        )

        self.task_def = ecs.FargateTaskDefinition(
            self, "EvalTaskDef",
            family="ti-edgeai-eval",
            cpu=2048,
            memory_limit_mib=4096,
        )
        self.task_def.add_container(
            "EvalContainer",
            image=ecs.ContainerImage.from_ecr_repository(self.ecr_repo, "latest"),
            logging=ecs.LogDrivers.aws_logs(stream_prefix="ti-edgeai-eval"),
            environment={"PLATFORM": "am62a"},
        )

        cdk.CfnOutput(self, "EcrRepoUri", value=self.ecr_repo.repository_uri)
        cdk.CfnOutput(self, "EvalClusterArn", value=self.cluster.cluster_arn)
