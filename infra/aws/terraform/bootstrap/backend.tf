# Added after the first apply (see README.md): the bootstrap's own state now
# lives in the bucket it created, so no state exists only on a laptop.
terraform {
  backend "s3" {
    bucket       = "ariadne-tfstate-836505524290"
    key          = "bootstrap/terraform.tfstate"
    region       = "eu-north-1"
    encrypt      = true
    use_lockfile = true
  }
}
