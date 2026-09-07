variable "project" {
  description = "Name prefix for every resource, so cost reports group cleanly."
  type        = string
  default     = "ariadne"
}

variable "region" {
  description = "AWS region. Must match wherever the Graviton benchmark instance lives; cross-region data transfer would distort cost-per-frame."
  type        = string
  default     = "us-east-1"
}

variable "artifact_bucket_name" {
  description = "Globally unique S3 bucket name for datasets, scenes, checkpoints and run artifacts (arch 28.1)."
  type        = string
}

variable "runs_ia_transition_days" {
  description = "Days before runs/ objects move to Infrequent Access (arch 27.3)."
  type        = number
  default     = 30
}

variable "tags" {
  description = "Tags applied to everything, so a competition budget can be attributed."
  type        = map(string)
  default = {
    Project   = "ariadne"
    ManagedBy = "terraform"
  }
}
