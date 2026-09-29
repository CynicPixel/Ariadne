variable "region" {
  description = "Region for the state bucket. Same as the workload region (eu-north-1), so state reads never cross regions."
  type        = string
  default     = "eu-north-1"
}
