# Artifact bucket (arch 28.1) with the lifecycle of 27.3.
#
# One bucket with prefixes rather than several: datasets/, scenes/,
# checkpoints/ and runs/ share a lifecycle policy surface and an access policy,
# and a single bucket keeps the S3 URI in a run manifest short and stable.

resource "aws_s3_bucket" "artifacts" {
  bucket = var.bucket_name
  tags   = merge(var.tags, { Name = "${var.project}-artifacts" })
}

# Versioning is on because run artifacts are evidence. An overwritten metrics
# file with no history is indistinguishable from a fabricated one, and 36
# requires a figure to be reproducible from its directory.
resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Nothing here is public. Recordings and reconstructions are of simulated
# interiors, but the habit matters and the responsible-use position in the
# report depends on it being true rather than asserted.
resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  # Run artifacts are read constantly for a few weeks, then only when a figure
  # is regenerated. IA is cheaper to store and more expensive to read, which
  # matches that access pattern.
  rule {
    id     = "runs-to-ia"
    status = "Enabled"
    filter {
      prefix = "runs/"
    }
    transition {
      days          = var.runs_ia_transition_days
      storage_class = "STANDARD_IA"
    }
  }

  # Bags are large and regenerable from a scene plus a seed, but regenerating
  # costs GPU time, so they are kept and simply moved down a tier.
  rule {
    id     = "datasets-to-ia"
    status = "Enabled"
    filter {
      prefix = "datasets/"
    }
    transition {
      days          = 60
      storage_class = "STANDARD_IA"
    }
  }

  # Old versions are a safety net, not an archive. Expire them so versioning
  # does not quietly become the largest line on the bill.
  rule {
    id     = "expire-noncurrent"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }

  # A failed multipart upload otherwise bills forever while being invisible in
  # the console object listing.
  rule {
    id     = "abort-incomplete-uploads"
    status = "Enabled"
    filter {}
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}
