# Pinned so a `terraform init` months from now resolves the same provider.
terraform {
  # >= 1.11: S3-native state locking (use_lockfile) is GA from 1.11, and the
  # DynamoDB lock table it replaces is deprecated (Terraform S3 backend docs).
  required_version = ">= 1.11"

  # State lives in the bucket created by ./bootstrap. Backend blocks cannot
  # read variables, so the account-specific name is written literally.
  backend "s3" {
    bucket       = "ariadne-tfstate-836505524290"
    key          = "main/terraform.tfstate"
    region       = "eu-north-1"
    encrypt      = true
    use_lockfile = true
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }
}
