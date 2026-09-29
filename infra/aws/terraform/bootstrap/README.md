# AWS bring-up runbook

Account `836505524290`, region `eu-north-1` (D28), **paid plan with $200 credits** (D26, D27).
Everything runs through `infra/scripts/aws.sh`; nothing is installed on the host.

## 0. Console, as root (once)

1. Enable MFA on the root user. Create **no** root access keys.
2. IAM -> Users -> create a user with console access, MFA, and `AdministratorAccess`.
3. Sign in once as that user at `https://836505524290.signin.aws.amazon.com/console`.

Never join or create an AWS Organization, and never set up Control Tower: on the free plan
that upgrades the account AND expires the free credits immediately.

## 1. CLI sign-in

```bash
infra/scripts/aws.sh login            # prints a URL; sign in as the IAM user; paste the code back
infra/scripts/aws.sh aws sts get-caller-identity   # must show the IAM user, not root
```

## 2. Bootstrap the state bucket (local state, once)

```bash
infra/scripts/aws.sh tf bootstrap init
infra/scripts/aws.sh tf bootstrap plan
infra/scripts/aws.sh tf bootstrap apply
```

Then move the bootstrap's own state into the bucket it created. Add
`infra/aws/terraform/bootstrap/backend.tf`:

```hcl
terraform {
  backend "s3" {
    bucket       = "ariadne-tfstate-836505524290"
    key          = "bootstrap/terraform.tfstate"
    region       = "eu-north-1"
    encrypt      = true
    use_lockfile = true
  }
}
```

```bash
infra/scripts/aws.sh tf bootstrap init -migrate-state
rm infra/aws/terraform/bootstrap/terraform.tfstate*   # only after the migration succeeds
```

## 3. Main stack

```bash
cp infra/aws/terraform/terraform.tfvars.example infra/aws/terraform/terraform.tfvars  # then edit
infra/scripts/aws.sh tf main init
infra/scripts/aws.sh tf main plan
infra/scripts/aws.sh tf main apply
```

Then in GitHub -> Settings -> Secrets and variables -> Actions -> **Variables** (not secrets):
`AWS_CI_ROLE_ARN` = the `github_ci_role_arn` output, `AWS_REGION` = `eu-north-1`.

## 4. Check the budget actually sees spend

Once anything billable has run, confirm the `ariadne-free-credits` budget shows non-zero actual
spend. It is configured with `include_credit = false`; if it still reads $0 on the free plan, the
alarm cannot protect the credits and something else must.
