"""MonitoringStack — CloudWatch alarms and dashboard for TI EdgeAI MLOps."""
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_cloudwatch as cw,
    aws_cloudwatch_actions as cw_actions,
    aws_sns as sns,
    aws_sns_subscriptions as subscriptions,
    Duration,
)
from constructs import Construct
from stacks.pipeline_stack import PipelineStack


class MonitoringStack(Stack):
    def __init__(self, scope: Construct, construct_id: str,
                 pipeline_stack: PipelineStack,
                 **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.alert_topic = sns.Topic(
            self, "AlertTopic",
            topic_name="ti-edgeai-alerts",
            display_name="TI EdgeAI MLOps Alerts",
        )
        alert_email = self.node.try_get_context("alert_email")
        if alert_email:
            self.alert_topic.add_subscription(subscriptions.EmailSubscription(alert_email))

        alarm_action = cw_actions.SnsAction(self.alert_topic)

        # Top-1 classification accuracy < 68% (registry gate threshold)
        self.accuracy_alarm = cw.Alarm(
            self, "AccuracyAlarm",
            alarm_name="ti-edgeai-classification-accuracy-low",
            alarm_description="Top-1 classification accuracy dropped below 68% gate",
            metric=cw.Metric(
                namespace="TIEdgeAI",
                metric_name="ModelAccuracy",
                dimensions_map={"Task": "classification", "Platform": "am62a"},
                period=Duration.minutes(5),
                statistic="Average",
            ),
            threshold=0.68,
            comparison_operator=cw.ComparisonOperator.LESS_THAN_THRESHOLD,
            evaluation_periods=2,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
        )
        self.accuracy_alarm.add_alarm_action(alarm_action)

        # p95 inference latency > 500ms
        self.latency_alarm = cw.Alarm(
            self, "LatencyAlarm",
            alarm_name="ti-edgeai-inference-latency-high",
            alarm_description="p95 inference latency exceeded 500ms threshold",
            metric=cw.Metric(
                namespace="TIEdgeAI",
                metric_name="InferenceLatencyP95",
                dimensions_map={"Platform": "am62a"},
                period=Duration.minutes(5),
                statistic="p95",
            ),
            threshold=500,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_THRESHOLD,
            evaluation_periods=2,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
        )
        self.latency_alarm.add_alarm_action(alarm_action)

        # Pipeline execution failure
        self.pipeline_failure_alarm = cw.Alarm(
            self, "PipelineFailureAlarm",
            alarm_name="ti-edgeai-pipeline-execution-failed",
            alarm_description="TI EdgeAI MLOps pipeline stage failed",
            metric=cw.Metric(
                namespace="AWS/CodePipeline",
                metric_name="FailedPipelineExecutions",
                dimensions_map={"PipelineName": pipeline_stack.pipeline.pipeline_name},
                period=Duration.minutes(5),
                statistic="Sum",
            ),
            threshold=1,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            evaluation_periods=1,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
        )
        self.pipeline_failure_alarm.add_alarm_action(alarm_action)

        self.dashboard = cw.Dashboard(
            self, "Dashboard",
            dashboard_name="TIEdgeAI-MLOps",
        )
        self.dashboard.add_widgets(
            cw.TextWidget(
                markdown="# TI EdgeAI Model Zoo MLOps Dashboard\nSoCs: TDA4VM, AM62A, AM68A, AM69A",
                width=24, height=2,
            ),
            cw.AlarmStatusWidget(
                title="Pipeline Health",
                alarms=[self.accuracy_alarm, self.latency_alarm, self.pipeline_failure_alarm],
                width=12, height=6,
            ),
            cw.GraphWidget(
                title="Classification Accuracy",
                left=[
                    cw.Metric(
                        namespace="TIEdgeAI",
                        metric_name="ModelAccuracy",
                        dimensions_map={"Task": "classification", "Platform": "am62a"},
                        label="Top-1 (AM62A)",
                        period=Duration.minutes(5),
                        statistic="Average",
                    ),
                    cw.Metric(
                        namespace="TIEdgeAI",
                        metric_name="ModelAccuracy",
                        dimensions_map={"Task": "classification", "Platform": "tda4vm"},
                        label="Top-1 (TDA4VM)",
                        period=Duration.minutes(5),
                        statistic="Average",
                    ),
                ],
                width=12, height=6,
            ),
            cw.GraphWidget(
                title="Inference Latency p95 (ms)",
                left=[
                    cw.Metric(
                        namespace="TIEdgeAI",
                        metric_name="InferenceLatencyP95",
                        dimensions_map={"Platform": "am62a"},
                        label="AM62A p95",
                        period=Duration.minutes(5),
                        statistic="p95",
                    ),
                ],
                width=12, height=6,
            ),
        )

        cdk.CfnOutput(self, "AlertTopicArn", value=self.alert_topic.topic_arn)
        cdk.CfnOutput(self, "DashboardName", value=self.dashboard.dashboard_name)
