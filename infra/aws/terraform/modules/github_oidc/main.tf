# ---------------------------------------------------------------------------
# GitHub Actions -> AWS without stored credentials (GitHub OIDC).
#
# GitHub mints a short-lived OIDC token per workflow run; AWS STS exchanges it
# for temporary role credentials if the token's claims match this role's trust
# policy. The repository therefore holds ZERO AWS secrets: the role ARN is a
# plain repository variable, useless to anyone whose token does not match.
#
# Works in a standalone account. It does not need AWS Organizations, which on
# the free plan would expire the free credits (see DECISIONS D26).
# ---------------------------------------------------------------------------

resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
  # thumbprint_list omitted: for this issuer IAM validates against its own
  # trusted CA library. If the pinned provider version rejects the omission,
  # `terraform validate` fails here rather than at apply.
  tags = var.tags
}

data "aws_iam_policy_document" "trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }
    # Audience: only tokens minted for AWS STS.
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    # Subject: which repository and ref may assume the role. GitHub requires
    # at least one such condition, since without it ANY repository's token
    # would be accepted.
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = var.allowed_subjects
    }
  }
}

resource "aws_iam_role" "ci" {
  name                 = "${var.project}-github-ci"
  assume_role_policy   = data.aws_iam_policy_document.trust.json
  max_session_duration = 3600
  tags                 = var.tags
}

# Least privilege: push images to THIS project's ECR repositories, nothing
# else. GetAuthorizationToken is account-wide by AWS's design (no resource
# scoping exists for it); every other action is pinned to the repo ARNs.
data "aws_iam_policy_document" "ecr_push" {
  statement {
    sid       = "EcrLogin"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid = "EcrPushPull"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:CompleteLayerUpload",
      "ecr:DescribeImages",
      "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
    ]
    resources = var.ecr_repository_arns
  }
}

resource "aws_iam_role_policy" "ecr_push" {
  name   = "ecr-push"
  role   = aws_iam_role.ci.id
  policy = data.aws_iam_policy_document.ecr_push.json
}
