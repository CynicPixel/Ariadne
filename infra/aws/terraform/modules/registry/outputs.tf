output "repository_urls" {
  value = { for k, v in aws_ecr_repository.images : k => v.repository_url }
}
