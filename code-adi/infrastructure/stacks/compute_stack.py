"""ComputeStack — ECR, ECS Fargate, and EC2 spot resources for ADI MLOps."""
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_ecr as ecr,
    aws_ecs as ecs,
    aws_ec2 as ec2,
    aws_iam as iam,
    RemovalPolicy,
    Duration,
)
from constructs import Construct


class ComputeStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ECR — AI8X training image (PyTorch + ai8x-training env)
        self.training_repo = ecr.Repository(
            self, "TrainingRepo",
            repository_name="adi-ai8x-training",
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                ecr.LifecycleRule(
                    description="Keep latest 5 training images",
                    max_image_count=5,
                    rule_priority=1,
                )
            ],
        )

        # ECR — ai8xize synthesis image
        self.synthesis_repo = ecr.Repository(
            self, "SynthesisRepo",
            repository_name="adi-ai8x-synthesis",
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                ecr.LifecycleRule(
                    description="Keep latest 5 synthesis images",
                    max_image_count=5,
                    rule_priority=1,
                )
            ],
        )

        # VPC — 2 AZs, 1 NAT gateway (evaluation workloads are not latency-sensitive)
        self.vpc = ec2.Vpc(
            self, "Vpc",
            vpc_name="adi-modelzoo-vpc",
            max_azs=2,
            nat_gateways=1,
        )

        # ECS cluster — Fargate for evaluation (CPU); spot EC2 for QAT training
        self.cluster = ecs.Cluster(
            self, "EvalCluster",
            cluster_name="adi-modelzoo-eval-cluster",
            vpc=self.vpc,
            container_insights=True,
        )

        # Fargate task definition for model evaluation
        self.eval_task_def = ecs.FargateTaskDefinition(
            self, "EvalTaskDef",
            family="adi-modelzoo-eval",
            cpu=2048,
            memory_limit_mib=4096,
        )
        self.eval_task_def.add_container(
            "EvalContainer",
            image=ecs.ContainerImage.from_ecr_repository(self.synthesis_repo, "latest"),
            logging=ecs.LogDrivers.aws_logs(stream_prefix="adi-eval"),
            environment={"PLATFORM": "max32690"},
        )

        # IAM instance profile for EC2 spot GPU QAT training instances
        self.spot_instance_role = iam.Role(
            self, "SpotInstanceRole",
            role_name="adi-modelzoo-spot-instance-role",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryReadOnly"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3ReadOnlyAccess"),
            ],
        )
        self.spot_instance_profile = iam.CfnInstanceProfile(
            self, "SpotInstanceProfile",
            roles=[self.spot_instance_role.role_name],
            instance_profile_name="adi-modelzoo-spot-profile",
        )

        # EC2 launch template — g4dn.xlarge spot for GPU QAT training
        # Uses Deep Learning Base AMI placeholder; override AMI at deployment
        self.training_launch_template = ec2.LaunchTemplate(
            self, "TrainingLaunchTemplate",
            launch_template_name="adi-ai8x-training-spot",
            instance_type=ec2.InstanceType("g4dn.xlarge"),
            machine_image=ec2.MachineImage.lookup(
                name="Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2) *",
                owners=["amazon"],
            ),
            spot_options=ec2.LaunchTemplateSpotOptions(
                request_type=ec2.SpotRequestType.ONE_TIME,
                max_price=0.40,  # g4dn.xlarge on-demand ~$0.526
            ),
            role=self.spot_instance_role,
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/xvda",
                    volume=ec2.BlockDeviceVolume.ebs(100, volume_type=ec2.EbsDeviceVolumeType.GP3),
                )
            ],
        )

        # Outputs
        cdk.CfnOutput(self, "TrainingRepoUri", value=self.training_repo.repository_uri)
        cdk.CfnOutput(self, "SynthesisRepoUri", value=self.synthesis_repo.repository_uri)
        cdk.CfnOutput(self, "EvalClusterArn", value=self.cluster.cluster_arn)
        cdk.CfnOutput(self, "TrainingLaunchTemplateId",
                      value=self.training_launch_template.launch_template_id or "")
