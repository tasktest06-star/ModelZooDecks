"""CloudWatch monitoring for NXP eIQ Model Zoo pipeline."""

import aws_cdk as cdk
from aws_cdk import (
    Stack, aws_cloudwatch as cloudwatch, aws_cloudwatch_actions as cw_actions,
    aws_sns as sns, Duration,
)
from constructs import Construct


class MonitoringStack(Stack):
    def __init__(self, scope, construct_id, artifact_bucket, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        self.alert_topic = sns.Topic(
            self, "AlertTopic",
            topic_name="nxp-modelzoo-alerts",
            display_name="NXP eIQ Model Zoo MLOps Alerts",
        )

        # MobileNetV2 classification accuracy on i.MX 8M Plus
        self.cls_accuracy_alarm = cloudwatch.Alarm(
            self, "ClassificationAccuracyAlarm",
            alarm_name="nxp-mobilenetv2-accuracy-drop",
            alarm_description="MobileNetV2 Top-1 accuracy dropped below 70% gate",
            metric=cloudwatch.Metric(
                namespace="NXPModelZoo/Evaluation",
                metric_name="Top1Accuracy",
                dimensions_map={"Model": "mobilenetv2", "Platform": "imx8mplus"},
                period=Duration.hours(1),
                statistic="Average",
            ),
            threshold=0.70,
            comparison_operator=cloudwatch.ComparisonOperator.LESS_THAN_THRESHOLD,
            evaluation_periods=2,
            treat_missing_data=cloudwatch.TreatMissingData.BREACHING,
        )
        self.cls_accuracy_alarm.add_alarm_action(cw_actions.SnsAction(self.alert_topic))

        # Fast-SRGAN PSNR gate (>= 28 dB)
        self.psnr_alarm = cloudwatch.Alarm(
            self, "PSNRAlarm",
            alarm_name="nxp-fast-srgan-psnr-drop",
            alarm_description="Fast-SRGAN PSNR dropped below 28 dB gate",
            metric=cloudwatch.Metric(
                namespace="NXPModelZoo/Evaluation",
                metric_name="PSNR_dB",
                dimensions_map={"Model": "fast_srgan", "Platform": "imx8mplus"},
                period=Duration.hours(1),
                statistic="Average",
            ),
            threshold=28.0,
            comparison_operator=cloudwatch.ComparisonOperator.LESS_THAN_THRESHOLD,
            evaluation_periods=2,
            treat_missing_data=cloudwatch.TreatMissingData.BREACHING,
        )
        self.psnr_alarm.add_alarm_action(cw_actions.SnsAction(self.alert_topic))

        # Inference latency p95 on i.MX 8M Plus (<= 150ms)
        self.latency_alarm = cloudwatch.Alarm(
            self, "LatencyAlarm",
            alarm_name="nxp-inference-latency-p95",
            alarm_description="i.MX 8M Plus inference latency p95 exceeded 150ms",
            metric=cloudwatch.Metric(
                namespace="NXPModelZoo/Inference",
                metric_name="InferenceLatencyMs",
                dimensions_map={"Platform": "imx8mplus"},
                period=Duration.minutes(5),
                statistic="p95",
            ),
            threshold=150.0,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            evaluation_periods=3,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        self.latency_alarm.add_alarm_action(cw_actions.SnsAction(self.alert_topic))

        self.dashboard = cloudwatch.Dashboard(
            self, "Dashboard",
            dashboard_name="NXPModelZoo-MLOps",
        )
        self.dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="MobileNetV2 Top-1 Accuracy (i.MX 8M Plus)",
                left=[cloudwatch.Metric(
                    namespace="NXPModelZoo/Evaluation",
                    metric_name="Top1Accuracy",
                    dimensions_map={"Model": "mobilenetv2"},
                    period=Duration.hours(1),
                )],
                width=12,
            ),
            cloudwatch.GraphWidget(
                title="Fast-SRGAN PSNR (dB)",
                left=[cloudwatch.Metric(
                    namespace="NXPModelZoo/Evaluation",
                    metric_name="PSNR_dB",
                    dimensions_map={"Model": "fast_srgan"},
                    period=Duration.hours(1),
                )],
                width=12,
            ),
            cloudwatch.AlarmStatusWidget(
                title="Alarm Status",
                alarms=[self.cls_accuracy_alarm, self.psnr_alarm, self.latency_alarm],
                width=24,
                height=3,
            ),
        )

        cdk.CfnOutput(self, "AlertTopicArn", value=self.alert_topic.topic_arn)
        cdk.CfnOutput(self, "DashboardName", value=self.dashboard.dashboard_name)
