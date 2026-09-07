# Ariadne infrastructure (arch 27.3).
#
# Scope here is network, storage and registry: the three that must exist before
# anything can be built or stored. Compute, sweeps and budget alarms follow in
# Phase 7 when there is something to sweep.
#
# Deliberately no EKS and no Kubernetes. The sweep pattern is SQS plus
# containers, which one person can operate under competition time pressure.

provider "aws" {
  region = var.region
  default_tags {
    tags = var.tags
  }
}

module "network" {
  source  = "./modules/network"
  project = var.project
  tags    = var.tags
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
