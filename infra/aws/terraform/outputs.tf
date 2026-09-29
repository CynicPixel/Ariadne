output "artifact_bucket" {
  value = module.storage.bucket_id
}

output "ecr_repository_urls" {
  description = "Push by digest, never by tag: run manifests reference digests (arch 27.2)."
  value       = module.registry.repository_urls
}

output "github_ci_role_arn" {
  description = "Set as GitHub repository VARIABLE AWS_CI_ROLE_ARN (not a secret)."
  value       = module.github_oidc.ci_role_arn
}

# Consumed by infra/scripts/spinup.sh (read via `terraform output -json`).
output "runner" {
  value = {
    instance_profile  = module.compute.instance_profile_name
    security_group_id = module.compute.security_group_id
    # AZ -> subnet: spinup.sh launches into whichever AZ has the cheapest
    # spot price at that moment.
    subnets_by_az   = { for s in data.aws_subnet.default : s.availability_zone => s.id }
    artifact_bucket = module.storage.bucket_id
  }
}
