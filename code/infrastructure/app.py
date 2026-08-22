#!/usr/bin/env python3
"""TI EdgeAI Model Zoo MLOps — CDK app entry point."""
import aws_cdk as cdk
from stacks.storage_stack import StorageStack
from stacks.compute_stack import ComputeStack
from stacks.pipeline_stack import PipelineStack
from stacks.monitoring_stack import MonitoringStack

app = cdk.App()
env = cdk.Environment(account=app.node.try_get_context("account") or "123456789012",
                      region=app.node.try_get_context("region") or "us-east-1")

storage = StorageStack(app, "TiStorageStack", env=env)
compute = ComputeStack(app, "TiComputeStack", env=env)
pipeline = PipelineStack(app, "TiPipelineStack",
                         storage_stack=storage,
                         compute_stack=compute,
                         env=env)
monitoring = MonitoringStack(app, "TiMonitoringStack",
                             pipeline_stack=pipeline,
                             env=env)

cdk.Tags.of(app).add("Project", "ti-edgeai-model-zoo")
cdk.Tags.of(app).add("ManagedBy", "CDK")

app.synth()
