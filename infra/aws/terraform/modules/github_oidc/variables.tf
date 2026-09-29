variable "project" {
  type = string
}

variable "allowed_subjects" {
  description = <<-EOT
    OIDC `sub` patterns allowed to assume the CI role (StringLike).
    GitHub repos created after 2026-07-15 use the IMMUTABLE format
    repo:OWNER@OWNER_ID/REPO@REPO_ID:ref:refs/heads/BRANCH. The owner id is
    pinned; the repo id is wildcarded only because the repository is private
    and its id could not be read. Tighten to the exact id once the first CI run
    reveals it (it appears in the CloudTrail AssumeRoleWithWebIdentity event).
  EOT
  type        = list(string)
}

variable "ecr_repository_arns" {
  type = list(string)
}

variable "tags" {
  type    = map(string)
  default = {}
}
