output "vpc_id" {
  value = module.network.vpc_id
}

output "public_subnet_id" {
  value = module.network.public_subnet_id
}

output "artifact_bucket" {
  value = module.storage.bucket_id
}

output "ecr_repository_urls" {
  description = "Push by digest, never by tag: run manifests reference digests (arch 27.2)."
  value       = module.registry.repository_urls
}
