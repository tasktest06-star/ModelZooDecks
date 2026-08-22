"""Unit tests for TI MonitoringStack."""
import pytest
import aws_cdk as cdk
from aws_cdk.assertions import Template, Match
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stacks.storage_stack import StorageStack
from stacks.compute_stack import ComputeStack
from stacks.pipeline_stack import PipelineStack
from stacks.monitoring_stack import MonitoringStack


@pytest.fixture(scope="module")
def template():
    app = cdk.App()
    env = cdk.Environment(account="123456789012", region="us-east-1")
    storage = StorageStack(app, "TestTiStorageForMonitor", env=env)
    compute = ComputeStack(app, "TestTiComputeForMonitor", env=env)
    pipeline = PipelineStack(app, "TestTiPipelineForMonitor",
                             storage_stack=storage,
                             compute_stack=compute,
                             env=env)
    monitoring = MonitoringStack(app, "TestTiMonitoringStack",
                                 pipeline_stack=pipeline,
                                 env=env)
    return Template.from_stack(monitoring)


def test_sns_topic_exists(template):
    template.has_resource_properties("AWS::SNS::Topic", {
        "TopicName": "ti-edgeai-alerts",
        "DisplayName": "TI EdgeAI MLOps Alerts",
    })


def test_accuracy_alarm_exists(template):
    template.has_resource_properties("AWS::CloudWatch::Alarm", {
        "AlarmName": "ti-edgeai-classification-accuracy-low",
        "Threshold": 0.68,
        "ComparisonOperator": "LessThanThreshold",
    })


def test_latency_alarm_exists(template):
    template.has_resource_properties("AWS::CloudWatch::Alarm", {
        "AlarmName": "ti-edgeai-inference-latency-high",
        "Threshold": 500,
        "ComparisonOperator": "GreaterThanThreshold",
    })


def test_pipeline_failure_alarm_exists(template):
    template.has_resource_properties("AWS::CloudWatch::Alarm", {
        "AlarmName": "ti-edgeai-pipeline-execution-failed",
        "Threshold": 1,
    })


def test_three_alarms_total(template):
    template.resource_count_is("AWS::CloudWatch::Alarm", 3)


def test_dashboard_exists(template):
    template.has_resource_properties("AWS::CloudWatch::Dashboard", {
        "DashboardName": "TIEdgeAI-MLOps",
    })
