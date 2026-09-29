variable "project" {
  description = "Name prefix for every resource, so cost reports group cleanly."
  type        = string
  default     = "ariadne"
}

variable "region" {
  description = "AWS region for everything. eu-north-1 (Stockholm) was chosen as the cheapest region with c8g, g4dn, COOL and Bedrock (DECISIONS D28). Keep ALL resources in one region: cross-region data transfer would distort cost-per-frame and spend credits for nothing."
  type        = string
  default     = "eu-north-1"
}

variable "artifact_bucket_name" {
  description = "Globally unique S3 bucket name for datasets, scenes, checkpoints and run artifacts (arch 28.1). S3 names must be lowercase; the account id makes it unique."
  type        = string
  default     = "ariadne-836505524290-eun1"
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

variable "github_ci_subjects" {
  description = "OIDC subject patterns allowed to assume the CI role. See modules/github_oidc/variables.tf for why the repo id is wildcarded."
  type        = list(string)
  default     = ["repo:CynicPixel@95163434/Ariadne@*:ref:refs/heads/main"]
}

variable "credit_budget_usd" {
  description = "Free-tier credit pool in USD. 100 until the credit-earning activities are complete, then 200."
  type        = number
  default     = 100
}

variable "alert_email" {
  description = "Budget alarm recipient. Set in terraform.tfvars (gitignored) so a personal address never lands in the repository."
  type        = string
}
