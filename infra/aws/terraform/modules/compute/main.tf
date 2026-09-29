# ---------------------------------------------------------------------------
# Ephemeral test compute: the identity and network envelope that
# infra/scripts/spinup.sh launches instances into (arch 27.3 "compute",
# reduced to what tests-only spin-ups need; see DECISIONS D27).
#
# No launch template and no always-on resource: instances exist only for the
# duration of one test run. Everything here is free while nothing is running.
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "ec2_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "runner" {
  name               = "${var.project}-test-runner"
  assume_role_policy = data.aws_iam_policy_document.ec2_trust.json
  tags               = var.tags
}

# Least privilege: read this run's inputs and write its results, under
# runs/ only. The instance cannot touch datasets/, scenes/ or anything else.
data "aws_iam_policy_document" "runner" {
  statement {
    sid       = "RunArtifacts"
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${var.artifact_bucket_arn}/runs/*"]
  }
  statement {
    sid       = "EcrLogin"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid = "EcrPull"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = var.ecr_repository_arns
  }
}

resource "aws_iam_role_policy" "runner" {
  name   = "run-artifacts-and-ecr-pull"
  role   = aws_iam_role.runner.id
  policy = data.aws_iam_policy_document.runner.json
}

# Debugging is done through SSM Session Manager, never SSH: no key pairs, no
# inbound port, nothing to leak.
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.runner.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "runner" {
  name = "${var.project}-test-runner"
  role = aws_iam_role.runner.name
  tags = var.tags
}

# Egress only. The instance pulls code/images and pushes results; nothing
# ever needs to connect TO it.
resource "aws_security_group" "runner" {
  name        = "${var.project}-test-runner"
  description = "Ephemeral test runners: egress only, no inbound"
  vpc_id      = var.vpc_id

  egress {
    description      = "All outbound (S3, ECR, package mirrors)"
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }

  tags = merge(var.tags, { Name = "${var.project}-test-runner" })
}
