#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Account-wide audit: is ANYTHING billing, in ANY region?
#
#   infra/scripts/aws_audit.sh          # table + list of anomalies; exit 1 if any
#
# Checks every enabled region for the resource classes that bill while idle
# or accumulate silently: instances, EBS volumes and snapshots, Elastic IPs,
# public IPv4 on network interfaces, NAT gateways, VPC endpoints, load
# balancers, RDS instances and snapshots, Lambda functions, CloudWatch log
# groups, customer-managed KMS keys, Secrets Manager secrets, and ECR repos.
# Then, globally: S3 bucket contents, ECR image counts, budget, credits.
#
# A failed query prints ERR, never 0. "Nothing found" and "could not look"
# must never look the same: during bring-up a shell bug made every query fail
# and an earlier version of this sweep would have read that as a clean bill.
# Expected steady state is recorded in EXPECTED below; anything else is flagged.
# ---------------------------------------------------------------------------
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AWS="$REPO_ROOT/infra/scripts/aws.sh"
HOME_REGION="${AWS_REGION:-eu-north-1}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

# Region -> expected non-zero cells ("column=value"). Everything else must be 0.
declare -A EXPECTED=( [eu-north-1]="ECR=5" )

COLS=(EC2 EBS snap EIP pubIP NAT VPCe ELB RDS RDSsnap Lambda logs CMK secrets ECR)
QUERIES=(
  "ec2 describe-instances --filters Name=instance-state-name,Values=pending,running,stopping,stopped --query length(Reservations[].Instances[])"
  "ec2 describe-volumes --query length(Volumes)"
  "ec2 describe-snapshots --owner-ids self --query length(Snapshots)"
  "ec2 describe-addresses --query length(Addresses)"
  "ec2 describe-network-interfaces --query length(NetworkInterfaces[?Association.PublicIp!=null])"
  "ec2 describe-nat-gateways --filter Name=state,Values=pending,available --query length(NatGateways)"
  "ec2 describe-vpc-endpoints --query length(VpcEndpoints)"
  "elbv2 describe-load-balancers --query length(LoadBalancers)"
  "rds describe-db-instances --query length(DBInstances)"
  "rds describe-db-snapshots --query length(DBSnapshots)"
  "lambda list-functions --query length(Functions)"
  "logs describe-log-groups --query length(logGroups)"
  "kms list-aliases --query length(Aliases[?!starts_with(AliasName,'alias/aws/')])"
  "secretsmanager list-secrets --query length(SecretList)"
  "ecr describe-repositories --query length(repositories)"
)

regions_raw="$("$AWS" aws ec2 describe-regions --query 'Regions[].RegionName' --output text 2>/dev/null)" \
  || { echo "cannot list regions (credentials?)" >&2; exit 2; }
read -r -a REGIONS <<< "$(echo "${regions_raw//$'\r'/}" | tr '\t' ' ')"

for R in "${REGIONS[@]}"; do
  (
    line="$R"
    for q in "${QUERIES[@]}"; do
      # shellcheck disable=SC2086  # q is a deliberately word-split argument list
      if out="$(AWS_REGION=$R "$AWS" aws $q --output text 2>/dev/null)"; then
        out="${out//$'\r'/}"; line+=$'\t'"${out%%$'\n'*}"
      else
        line+=$'\t'ERR
      fi
    done
    # One file per region, written once: parallel appends interleave.
    printf '%s\n' "$line" > "$TMP/$R"
  ) &
done
wait

{ printf 'region'; printf '\t%s' "${COLS[@]}"; printf '\n'; cat "$TMP"/* | sort; } | column -t

anomalies=0
echo
echo "=== anomalies (anything non-zero that is not in EXPECTED, and every ERR) ==="
while IFS=$'\t' read -r -a f; do
  r="${f[0]}"
  for i in "${!COLS[@]}"; do
    v="${f[$((i + 1))]}"; c="${COLS[$i]}"
    [ "$v" = 0 ] && continue
    case " ${EXPECTED[$r]:-} " in *" $c=$v "*) continue ;; esac
    echo "  $r  $c = $v"; anomalies=$((anomalies + 1))
  done
done < <(cat "$TMP"/*)
[ "$anomalies" -eq 0 ] && echo "  none"

echo
echo "=== global ==="
for b in $("$AWS" aws s3api list-buckets --query 'Buckets[].Name' --output text 2>/dev/null | tr -d '\r'); do
  printf '  s3 %-34s %s\n' "$b" \
    "$("$AWS" aws s3 ls "s3://$b" --recursive --summarize 2>/dev/null | tr -d '\r' | grep -E 'Total (Objects|Size)' | tr '\n' ' ')"
done
for r in $(AWS_REGION=$HOME_REGION "$AWS" aws ecr describe-repositories --query 'repositories[].repositoryName' --output text 2>/dev/null | tr -d '\r'); do
  printf '  ecr %-30s images=%s\n' "$r" \
    "$(AWS_REGION=$HOME_REGION "$AWS" aws ecr list-images --repository-name "$r" --query 'length(imageIds)' --output text 2>/dev/null | tr -d '\r')"
done
printf '  credits: %s\n' "$("$AWS" aws freetier get-account-plan-state --region us-east-1 \
  --query '[accountPlanType,accountPlanRemainingCredits.amount]' --output text 2>/dev/null | tr -d '\r')"

[ "$anomalies" -eq 0 ]
