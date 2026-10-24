# ECR repositories, one per image in the 27.1 table.
#
# Images are referenced BY DIGEST in run manifests (27.2). Tags are mutable
# pointers, so "ariadne-core:latest" identifies nothing reproducible.

locals {
  images = [
    "ariadne-core",         # dev, CI, Graviton stock baseline
    "ariadne-cool",         # arm64, linked against /opt/cool
    "ariadne-sim",          # Gazebo, amd64 + GPU
    "ariadne-gpu",          # PyTorch + gsplat reconstruction
    "ariadne-console",      # operator console
    "ariadne-orchestrator", # 7.x agentic orchestrator (Bedrock client) for live runs
  ]
}

resource "aws_ecr_repository" "images" {
  for_each = toset(local.images)

  name = "${var.project}/${each.value}"

  # Tags stay mutable: CI overwrites :latest for convenience. This is safe only
  # because manifests pin digests; if that ever stops being true, set this to
  # IMMUTABLE.
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = merge(var.tags, { Name = each.value })
}

# Untagged layers accumulate on every rebuild of the expensive opencv-base
# image and are pure cost once nothing references them.
resource "aws_ecr_lifecycle_policy" "expire_untagged" {
  for_each   = aws_ecr_repository.images
  repository = each.value.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Expire untagged images after 14 days"
      selection = {
        tagStatus   = "untagged"
        countType   = "sinceImagePushed"
        countUnit   = "days"
        countNumber = 14
      }
      action = { type = "expire" }
    }]
  })
}
