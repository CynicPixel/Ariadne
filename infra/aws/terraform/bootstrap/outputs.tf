output "state_bucket" {
  description = "Put this in the main stack's backend block (infra/aws/terraform/versions.tf)."
  value       = aws_s3_bucket.state.id
}

output "account_id" {
  value = data.aws_caller_identity.current.account_id
}
