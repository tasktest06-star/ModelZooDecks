"""MonitoringStack — CloudWatch alarms and dashboard for ADI Model Zoo MLOps."""
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
class MonitoringStack(Stack):
    def __init__(self, scope: Construct, construct_id: str,
                 pipeline_name: str = "adi-modelzoo-mlops",
                 qat_project_name: str = "adi-modelzoo-qat-train",
                 synthesis_project_name: str = "adi-modelzoo-synthesize",
                 **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # SNS topic for all ADI model-zoo alerts
        self.alert_topic = sns.Topic(
            self, "AlertTopic",
            topic_name="adi-modelzoo-alerts",
            display_name="ADI ModelZoo MLOps Alerts",
        )
        # Add email subscription via context (optional)
        alert_email = self.node.try_get_context("alert_email")
        if alert_email:
            self.alert_topic.add_subscription(subscriptions.EmailSubscription(alert_email))

        alarm_action = cw_actions.SnsAction(self.alert_topic)

        # Alarm 1 — feature_pyramid_net object detection mAP < 0.45
        self.detection_map_alarm = cw.Alarm(
            self, "DetectionMAPAlarm",
            alarm_name="adi-feature-pyramid-net-map-low",
            alarm_description="feature_pyramid_net mAP@0.5 dropped below gate 0.45 on MAX78000",
            metric=cw.Metric(
                namespace="ADIModelZoo",
                metric_name="ModelAccuracy",
                dimensions_map={
                    "ModelId": "feature_pyramid_net",
                    "Platform": "max78000",
                    "Task": "detection",
                },
                period=Duration.minutes(5),
                statistic="Average",
            ),
            threshold=0.45,
            comparison_operator=cw.ComparisonOperator.LESS_THAN_THRESHOLD,
            evaluation_periods=2,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
        )
        self.detection_map_alarm.add_alarm_action(alarm_action)

        # Alarm 2 — ds_cnn keyword spotting accuracy < 0.90
        self.kws_accuracy_alarm = cw.Alarm(
            self, "KWSAccuracyAlarm",
            alarm_name="adi-ds-cnn-kws-accuracy-low",
            alarm_description="ds_cnn KWS Top-1 accuracy dropped below gate 0.90 on MAX78000",
            metric=cw.Metric(
                namespace="ADIModelZoo",
                metric_name="ModelAccuracy",
                dimensions_map={
                    "ModelId": "ds_cnn",
                    "Platform": "max78000",
                    "Task": "kws",
                },
                period=Duration.minutes(5),
                statistic="Average",
            ),
            threshold=0.90,
            comparison_operator=cw.ComparisonOperator.LESS_THAN_THRESHOLD,
            evaluation_periods=2,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
        )
        self.kws_accuracy_alarm.add_alarm_action(alarm_action)

        # Alarm 3 — ai8xize synthesis failure rate > 10%
        self.synthesis_failure_alarm = cw.Alarm(
            self, "SynthesisFailureAlarm",
            alarm_name="adi-synthesis-failure-rate-high",
            alarm_description="ai8xize synthesis failure rate exceeds 10%",
            metric=cw.Metric(
                namespace="ADIModelZoo",
                metric_name="SynthesisFailureRate",
                dimensions_map={"Stage": "ai8xize"},
                period=Duration.minutes(10),
                statistic="Average",
            ),
            threshold=0.10,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_THRESHOLD,
            evaluation_periods=1,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
        )
        self.synthesis_failure_alarm.add_alarm_action(alarm_action)

        # Alarm 4 — pipeline execution failures
        self.pipeline_failure_alarm = cw.Alarm(
            self, "PipelineFailureAlarm",
            alarm_name="adi-pipeline-execution-failed",
            alarm_description="ADI MLOps pipeline stage failed",
            metric=cw.Metric(
                namespace="AWS/CodePipeline",
                metric_name="FailedPipelineExecutions",
                dimensions_map={"PipelineName": pipeline_name},
                period=Duration.minutes(5),
                statistic="Sum",
            ),
            threshold=1,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            evaluation_periods=1,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
        )
        self.pipeline_failure_alarm.add_alarm_action(alarm_action)

        # CloudWatch Dashboard
        self.dashboard = cw.Dashboard(
            self, "Dashboard",
            dashboard_name="ADIModelZoo-MLOps",
        )
        self.dashboard.add_widgets(
            cw.TextWidget(
                markdown="# ADI Model Zoo MLOps Dashboard\nPlatforms: MAX78000, MAX78002, MAX32690, ADSP-SC835",
                width=24,
                height=2,
            ),
            cw.AlarmStatusWidget(
                title="Pipeline Health",
                alarms=[
                    self.detection_map_alarm,
                    self.kws_accuracy_alarm,
                    self.synthesis_failure_alarm,
                    self.pipeline_failure_alarm,
                ],
                width=12,
                height=6,
            ),
            cw.GraphWidget(
                title="Model Accuracy by Task",
                left=[
                    cw.Metric(
                        namespace="ADIModelZoo",
                        metric_name="ModelAccuracy",
                        dimensions_map={"ModelId": "feature_pyramid_net", "Platform": "max78000", "Task": "detection"},
                        label="FPN mAP (MAX78000)",
                        period=Duration.minutes(5),
                        statistic="Average",
                    ),
                    cw.Metric(
                        namespace="ADIModelZoo",
                        metric_name="ModelAccuracy",
                        dimensions_map={"ModelId": "ds_cnn", "Platform": "max78000", "Task": "kws"},
                        label="DS-CNN KWS (MAX78000)",
                        period=Duration.minutes(5),
                        statistic="Average",
                    ),
                    cw.Metric(
                        namespace="ADIModelZoo",
                        metric_name="ModelAccuracy",
                        dimensions_map={"ModelId": "mobilenetv2_075", "Platform": "max32690", "Task": "classification"},
                        label="MobileNetV2 Top-1 (MAX32690)",
                        period=Duration.minutes(5),
                        statistic="Average",
                    ),
                ],
                width=12,
                height=6,
            ),
            cw.GraphWidget(
                title="Pipeline Build Duration",
                left=[
                    cw.Metric(
                        namespace="AWS/CodeBuild",
                        metric_name="Duration",
                        dimensions_map={"ProjectName": qat_project_name},
                        label="QAT Train Duration",
                        period=Duration.minutes(60),
                        statistic="Average",
                    ),
                    cw.Metric(
                        namespace="AWS/CodeBuild",
                        metric_name="Duration",
                        dimensions_map={"ProjectName": synthesis_project_name},
                        label="Synthesis Duration",
                        period=Duration.minutes(60),
                        statistic="Average",
                    ),
                ],
                width=12,
                height=6,
            ),
            cw.GraphWidget(
                title="Synthesis Failure Rate",
                left=[
                    cw.Metric(
                        namespace="ADIModelZoo",
                        metric_name="SynthesisFailureRate",
                        dimensions_map={"Stage": "ai8xize"},
                        label="ai8xize Failure Rate",
                        period=Duration.minutes(10),
                        statistic="Average",
                    ),
                ],
                width=12,
                height=6,
            ),
        )

        cdk.CfnOutput(self, "AlertTopicArn", value=self.alert_topic.topic_arn)
        cdk.CfnOutput(self, "DashboardName", value=self.dashboard.dashboard_name)
