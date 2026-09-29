# ---------------------------------------------------------------------------
# Credit budget alarms (arch 27.3, pulled forward from Phase 7 by D26).
#
# On the AWS free plan the account CLOSES when the credits run out, taking the
# recorded bags, images and the judging console with it. So this is not a
# nicety.
#
# TRAP: a cost budget includes credits by default, and credits net the cost
# down to $0, so the alarm would never fire. include_credit = false tracks
# GROSS spend, which is exactly what drains the credits.
#
# This is the BACKSTOP, not the brake. Budget data lags by hours; on a GPU
# instance real money is gone before an email arrives. The brake is on the
# instances themselves (terminate-on-shutdown + max lifetime), see D26.
#
# Monitoring budgets are free (AWS Budgets pricing page).
# ---------------------------------------------------------------------------

resource "aws_budgets_budget" "credits" {
  name         = "${var.project}-free-credits"
  budget_type  = "COST"
  limit_amount = format("%.2f", var.limit_usd)
  limit_unit   = "USD"
  # One window covering the whole credit lifetime, not a monthly reset:
  # the credits are a single pool, not a monthly allowance.
  time_unit = "ANNUALLY"

  cost_types {
    include_credit = false
    include_refund = false
  }

  dynamic "notification" {
    for_each = var.actual_thresholds_pct
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value
      threshold_type             = "PERCENTAGE"
      notification_type          = "ACTUAL"
      subscriber_email_addresses = [var.alert_email]
    }
  }

  # Forecast alarm: fires on the TREND, earlier than any actual threshold.
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.alert_email]
  }
}
