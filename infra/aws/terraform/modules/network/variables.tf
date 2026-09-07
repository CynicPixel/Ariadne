variable "project" { type = string }
variable "tags" { type = map(string) }

variable "allowed_console_cidrs" {
  description = "CIDRs allowed to reach the operator console. Empty by default: the console is closed until someone opens it deliberately, rather than exposed by accident."
  type        = list(string)
  default     = []
}
