variable "project" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "artifact_bucket_arn" {
  type = string
}

variable "ecr_repository_arns" {
  type = list(string)
}

variable "tags" {
  type    = map(string)
  default = {}
}
