# Ariadne infrastructure (arch 27.3).
#
# Scope: storage, registry, CI identity (GitHub OIDC), the ephemeral test
# runner envelope and the credit budget, on the region's default VPC. Sweeps
# follow in Phase 7. The budget was pulled forward
# from Phase 7 because on the free plan the account closes when credits run
# out (DECISIONS D26).
#
# Deliberately no EKS and no Kubernetes. The sweep pattern is SQS plus
# containers, which one person can operate under competition time pressure.

provider "aws" {
  region = var.region
  default_tags {
    tags = var.tags
  }
}

# NETWORK: the region's DEFAULT VPC, not one of our own (DECISIONS D29).
# Every region ships with a default VPC (public subnet per AZ, internet
# gateway, routes), already free and already there. Test runners need exactly
# that and nothing more, so creating a parallel VPC would add resources without
# adding capability. The only network object Ariadne creates is the runner's
# egress-only security group (modules/compute). modules/network is kept, unused,
# for the Phase 7 hosted console, which may justify its own VPC.
#
# Deliberately NOT used: NAT gateways (~$0.045/h each, the classic silent cost),
# Elastic IPs, interface endpoints. Instances reach S3/ECR over the internet
# gateway using their auto-assigned public IPv4, which bills ~$0.005/h only
# while an instance runs.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

data "aws_subnet" "default" {
  for_each = toset(data.aws_subnets.default.ids)
  id       = each.value
}

module "storage" {
  source                  = "./modules/storage"
  project                 = var.project
  bucket_name             = var.artifact_bucket_name
  runs_ia_transition_days = var.runs_ia_transition_days
  tags                    = var.tags
}

module "registry" {
  source  = "./modules/registry"
  project = var.project
  tags    = var.tags
}

module "github_oidc" {
  source              = "./modules/github_oidc"
  project             = var.project
  allowed_subjects    = var.github_ci_subjects
  ecr_repository_arns = values(module.registry.repository_arns)
  tags                = var.tags
}

module "budget" {
  source      = "./modules/budget"
  project     = var.project
  limit_usd   = var.credit_budget_usd
  alert_email = var.alert_email
}

module "compute" {
  source              = "./modules/compute"
  project             = var.project
  vpc_id              = data.aws_vpc.default.id
  artifact_bucket_arn = module.storage.bucket_arn
  ecr_repository_arns = values(module.registry.repository_arns)
  tags                = var.tags
}
