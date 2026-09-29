output "ci_role_arn" {
  description = "Set as the GitHub repository VARIABLE (not secret) AWS_CI_ROLE_ARN."
  value       = aws_iam_role.ci.arn
}
