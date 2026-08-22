import pytest
import aws_cdk as cdk
from aws_cdk.assertions import Template, Match
from stacks.storage_stack import StorageStack
from stacks.monitoring_stack import MonitoringStack


@pytest.fixture
def template():
    app = cdk.App()
    env = cdk.Environment(account="123456789012", region="us-east-1")
    storage = StorageStack(app, "S", env=env)
    mon = MonitoringStack(app, "M", artifact_bucket=storage.artifact_bucket, env=env)
    return Template.from_stack(mon)


class TestNXPMonitoringStack:
    def test_sns_topic(self, template):
        template.has_resource_properties("AWS::SNS::Topic", {
            "TopicName": "nxp-modelzoo-alerts",
        })

    def test_three_alarms(self, template):
        template.resource_count_is("AWS::CloudWatch::Alarm", 3)

    def test_cls_accuracy_alarm(self, template):
        template.has_resource_properties("AWS::CloudWatch::Alarm", {
            "AlarmName": "nxp-mobilenetv2-accuracy-drop",
            "Threshold": 0.70,
        })

    def test_psnr_alarm(self, template):
        template.has_resource_properties("AWS::CloudWatch::Alarm", {
            "AlarmName": "nxp-fast-srgan-psnr-drop",
            "Threshold": 28.0,
        })

    def test_latency_alarm(self, template):
        template.has_resource_properties("AWS::CloudWatch::Alarm", {
            "AlarmName": "nxp-inference-latency-p95",
            "Threshold": 150.0,
        })

    def test_dashboard(self, template):
        template.has_resource_properties("AWS::CloudWatch::Dashboard", {
            "DashboardName": "NXPModelZoo-MLOps",
        })

    def test_outputs(self, template):
        template.has_output("AlertTopicArn", {})
        template.has_output("DashboardName", {})
