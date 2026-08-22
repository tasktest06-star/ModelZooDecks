#!/usr/bin/env python3
"""ADI Model Zoo MLOps — CDK app entry point."""
import aws_cdk as cdk
from stacks.storage_stack import StorageStack
from stacks.compute_stack import ComputeStack
from stacks.pipeline_stack import PipelineStack
from stacks.monitoring_stack import MonitoringStack

app = cdk.App()
env = cdk.Environment(account=app.node.try_get_context("account") or "123456789012",
                      region=app.node.try_get_context("region") or "us-east-1")

storage = StorageStack(app, "AdiStorageStack", env=env)
compute = ComputeStack(app, "AdiComputeStack", env=env)
pipeline = PipelineStack(app, "AdiPipelineStack",
                         weights_bucket=storage.weights_bucket,
                         env=env)
monitoring = MonitoringStack(app, "AdiMonitoringStack", env=env)

cdk.Tags.of(app).add("Project", "adi-model-zoo")
cdk.Tags.of(app).add("ManagedBy", "CDK")

app.synth()
