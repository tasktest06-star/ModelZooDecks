import aws_cdk as cdk
from stacks.storage_stack import StorageStack
from stacks.compute_stack import ComputeStack
from stacks.pipeline_stack import PipelineStack
from stacks.monitoring_stack import MonitoringStack

app = cdk.App()
env = cdk.Environment(
    account=app.node.try_get_context("account") or "123456789012",
    region=app.node.try_get_context("region") or "us-east-1",
)

storage = StorageStack(app, "NXPModelZooStorage", env=env)
compute = ComputeStack(app, "NXPModelZooCompute", env=env)
pipeline = PipelineStack(app, "NXPModelZooPipeline",
                         tflite_bucket=storage.tflite_bucket,
                         vela_bucket=storage.vela_bucket,
                         artifact_bucket=storage.artifact_bucket,
                         env=env)
monitoring = MonitoringStack(app, "NXPModelZooMonitoring",
                              artifact_bucket=storage.artifact_bucket,
                              env=env)
app.synth()
