# ---------------------------------------------------------------------------
# Bootstrap: the ONE bucket that holds every other stack's Terraform state.
#
# Chicken-and-egg: the main stack's S3 backend needs this bucket to exist
# before `terraform init`. So this stack is applied once with LOCAL state, then
# migrated into the bucket it just created (see README.md in this directory).
# It is deliberately tiny, since it is the only stack ever run on local state.
#
# Free-plan note: S3 standard storage for a few KB of state is effectively free,
# and nothing here bills by the hour.
# ---------------------------------------------------------------------------
terraform {
  required_version = ">= 1.11"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "ariadne", ManagedBy = "terraform", Stack = "bootstrap" }
  }
}

data "aws_caller_identity" "current" {}

locals {
  # Account id in the name makes it globally unique without guessing.
  bucket_name = "ariadne-tfstate-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket" "state" {
  bucket = local.bucket_name

  # Losing this bucket loses the record of what exists, which on a
  # credit-capped account means orphaned resources quietly burning credits.
  lifecycle {
    prevent_destroy = true
  }
}

# Terraform's S3 backend docs: "It is highly recommended that you enable
# Bucket Versioning ... to allow for state recovery".
resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "state" {
  bucket = aws_s3_bucket.state.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "state" {
  bucket                  = aws_s3_bucket.state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# State can contain resource attributes worth protecting; refuse plaintext HTTP.
resource "aws_s3_bucket_policy" "state_tls_only" {
  bucket = aws_s3_bucket.state.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.state.arn, "${aws_s3_bucket.state.arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
  depends_on = [aws_s3_bucket_public_access_block.state]
}

# Keep old state versions long enough to recover from a bad apply, then let
# them go so versioning does not accumulate indefinitely.
resource "aws_s3_bucket_lifecycle_configuration" "state" {
  bucket = aws_s3_bucket.state.id
  rule {
    id     = "expire-noncurrent-state"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
  depends_on = [aws_s3_bucket_versioning.state]
}
