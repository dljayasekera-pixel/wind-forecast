# Terraform sketch of the AWS deployment. It shows how the pieces map to services
# and is NOT applied or validated: networking, IAM policies and the monitor Lambda
# are left as variables. Project B of this portfolio carries the full deployment.
#
#   S3            raw / curated / artifacts zones (versioned: data and model lineage)
#   Lambda        hourly NWP ingest, monitoring
#   ECS Fargate   prepare + predict (same container image as CI builds)
#   SageMaker     retraining job
#   Step Functions orchestration (statemachine.asl.json)
#   EventBridge   hourly schedule
#   CloudWatch    metrics published by windcast.monitoring, alarms below
#   EC2           not used: nothing here needs a long-lived host. A self-hosted
#                 Streamlit dashboard or a GRIB-decoding worker would be the case for it.

terraform {
  required_providers { aws = { source = "hashicorp/aws", version = "~> 5.0" } }
}

variable "project" { default = "windcast" }
variable "site" { default = "kelmarsh" }
variable "image_uri" { description = "ECR image built by CI" }
variable "subnets" { type = list(string) }
variable "ecs_cluster_arn" {}
variable "task_execution_role_arn" {}
variable "task_role_arn" {}
variable "sfn_role_arn" {}
variable "sagemaker_role_arn" {}
variable "lambda_role_arn" {}
variable "ingest_lambda_zip" {}
variable "monitor_lambda_arn" {}
variable "alert_email" {}

resource "aws_s3_bucket" "zone" {
  for_each = toset(["raw", "curated", "artifacts"])
  bucket   = "${var.project}-${var.site}-${each.key}"
}

resource "aws_s3_bucket_versioning" "zone" {
  for_each = aws_s3_bucket.zone
  bucket   = each.value.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_sns_topic" "alerts" { name = "${var.project}-alerts" }

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_lambda_function" "ingest" {
  function_name = "${var.project}-ingest-nwp"
  role          = var.lambda_role_arn
  runtime       = "python3.13"
  handler       = "lambda_ingest.handler"
  filename      = var.ingest_lambda_zip
  timeout       = 300
  memory_size   = 1024
  environment {
    variables = { RAW_BUCKET = aws_s3_bucket.zone["raw"].bucket, CONFIG_PATH = "configs/kelmarsh.yaml" }
  }
}

resource "aws_cloudwatch_log_group" "task" {
  name              = "/ecs/${var.project}"
  retention_in_days = 30
}

resource "aws_ecs_task_definition" "windcast" {
  family                   = var.project
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 2048
  memory                   = 8192
  execution_role_arn       = var.task_execution_role_arn
  task_role_arn            = var.task_role_arn
  container_definitions = jsonencode([{
    name      = "windcast"
    image     = var.image_uri
    essential = true
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.task.name
        "awslogs-region"        = "us-east-1"
        "awslogs-stream-prefix" = "task"
      }
    }
  }])
}

resource "aws_sfn_state_machine" "cycle" {
  name     = "${var.project}-forecast-cycle"
  role_arn = var.sfn_role_arn
  definition = templatefile("${path.module}/statemachine.asl.json", {
    ingest_lambda_arn   = aws_lambda_function.ingest.arn
    monitor_lambda_arn  = var.monitor_lambda_arn
    ecs_cluster_arn     = var.ecs_cluster_arn
    task_definition_arn = aws_ecs_task_definition.windcast.arn
    subnets_json        = jsonencode(var.subnets)
    config_path         = "configs/kelmarsh.yaml"
    sagemaker_role_arn  = var.sagemaker_role_arn
    image_uri           = var.image_uri
    artifact_bucket     = aws_s3_bucket.zone["artifacts"].bucket
    alert_topic_arn     = aws_sns_topic.alerts.arn
  })
}

resource "aws_cloudwatch_event_rule" "hourly" {
  name                = "${var.project}-hourly"
  schedule_expression = "cron(10 * * * ? *)"
}

resource "aws_cloudwatch_event_target" "cycle" {
  rule     = aws_cloudwatch_event_rule.hourly.name
  arn      = aws_sfn_state_machine.cycle.arn
  role_arn = var.sfn_role_arn
}

# Alarms on the metrics that windcast.monitoring.cloudwatch publishes (namespace WindCast).
locals {
  alarms = {
    nmae_high       = { metric = "nmae", threshold = 0.12, op = "GreaterThanThreshold", periods = 6 }
    scada_stale     = { metric = "scada_lag_h", threshold = 3, op = "GreaterThanThreshold", periods = 2 }
    coverage_low    = { metric = "interval_coverage", threshold = 0.80, op = "LessThanThreshold", periods = 24 }
    fallback_active = { metric = "fallback_share", threshold = 0.25, op = "GreaterThanThreshold", periods = 3 }
  }
}

resource "aws_cloudwatch_metric_alarm" "forecast" {
  for_each            = local.alarms
  alarm_name          = "${var.project}-${var.site}-${each.key}"
  namespace           = "WindCast"
  metric_name         = each.value.metric
  dimensions          = { Site = var.site }
  statistic           = "Average"
  period              = 3600
  evaluation_periods  = each.value.periods
  threshold           = each.value.threshold
  comparison_operator = each.value.op
  treat_missing_data  = "breaching" # no metric means the cycle is not running
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "cycle_failed" {
  alarm_name          = "${var.project}-cycle-failed"
  namespace           = "AWS/States"
  metric_name         = "ExecutionsFailed"
  dimensions          = { StateMachineArn = aws_sfn_state_machine.cycle.arn }
  statistic           = "Sum"
  period              = 3600
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}
