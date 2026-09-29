variable "project" {
  type = string
}

variable "limit_usd" {
  description = "Credit pool in USD. $100 at signup; raise to 200 once the credit-earning activities are done."
  type        = number
}

variable "actual_thresholds_pct" {
  description = "Actual-spend alarm thresholds, % of limit (arch 27.3 names 50 and 80; 25 added as an early canary)."
  type        = list(number)
  default     = [25, 50, 80]
}

variable "alert_email" {
  type = string
}
