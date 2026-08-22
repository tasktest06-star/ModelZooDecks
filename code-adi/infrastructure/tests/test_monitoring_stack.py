"""Unit tests for ADI MonitoringStack."""
import pytest
import aws_cdk as cdk
from aws_cdk.assertions import Template, Match
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stacks.monitoring_stack import MonitoringStack


@pytest.fixture(scope="module")
def template():
    app = cdk.App()
    env = cdk.Environment(account="123456789012", region="us-east-1")
    monitoring = MonitoringStack(app, "TestAdiMonitoringStack", env=env)
    return Template.from_stack(monitoring)


def test_sns_alert_topic_exists(template):
    template.has_resource_properties("AWS::SNS::Topic", {
        "TopicName": "adi-modelzoo-alerts",
        "DisplayName": "ADI ModelZoo MLOps Alerts",
    })


def test_detection_map_alarm_exists(template):
    template.has_resource_properties("AWS::CloudWatch::Alarm", {
        "AlarmName": "adi-feature-pyramid-net-map-low",
        "Threshold": 0.45,
        "ComparisonOperator": "LessThanThreshold",
    })


def test_kws_accuracy_alarm_exists(template):
    template.has_resource_properties("AWS::CloudWatch::Alarm", {
        "AlarmName": "adi-ds-cnn-kws-accuracy-low",
        "Threshold": 0.90,
        "ComparisonOperator": "LessThanThreshold",
    })


def test_synthesis_failure_alarm_exists(template):
    template.has_resource_properties("AWS::CloudWatch::Alarm", {
        "AlarmName": "adi-synthesis-failure-rate-high",
        "Threshold": 0.10,
        "ComparisonOperator": "GreaterThanThreshold",
    })


def test_pipeline_failure_alarm_exists(template):
    template.has_resource_properties("AWS::CloudWatch::Alarm", {
        "AlarmName": "adi-pipeline-execution-failed",
        "Threshold": 1,
        "ComparisonOperator": "GreaterThanOrEqualToThreshold",
    })


def test_dashboard_exists(template):
    template.has_resource_properties("AWS::CloudWatch::Dashboard", {
        "DashboardName": "ADIModelZoo-MLOps",
    })


def test_four_alarms_total(template):
    template.resource_count_is("AWS::CloudWatch::Alarm", 4)
