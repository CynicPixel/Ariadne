# One VPC with a single public subnet (arch 27.3).
#
# No NAT gateway and no private subnets: a NAT gateway costs roughly $32/month
# standing plus data processing, and nothing here needs egress from a private
# subnet. Instances that need the internet sit in the public subnet with a
# public IP. This is a competition project with a credit budget, not a
# production VPC, and the simpler topology is also the one person can debug.

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = merge(var.tags, { Name = "${var.project}-vpc" })
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = merge(var.tags, { Name = "${var.project}-igw" })
}

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true
  tags                    = merge(var.tags, { Name = "${var.project}-public" })
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  tags = merge(var.tags, { Name = "${var.project}-public-rt" })
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# Workers need egress (pull images, fetch bags, upload artifacts) but accept no
# inbound connections. Access is via SSM Session Manager rather than an open
# SSH port, so there is no key to leak and no port to scan.
resource "aws_security_group" "worker" {
  name        = "${var.project}-worker"
  description = "Ariadne compute: egress only, no inbound"
  vpc_id      = aws_vpc.main.id

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, { Name = "${var.project}-worker-sg" })
}

# The operator console (arch 29) is the one thing that serves traffic. Kept
# separate from the worker group so opening a port here can never widen access
# to the compute fleet. allowed_console_cidrs defaults to empty, so this group
# is closed until someone deliberately opens it.
resource "aws_security_group" "console" {
  name        = "${var.project}-console"
  description = "Operator console HTTP/WebSocket"
  vpc_id      = aws_vpc.main.id

  dynamic "ingress" {
    for_each = length(var.allowed_console_cidrs) > 0 ? [1] : []
    content {
      description = "Console HTTP"
      from_port   = 8000
      to_port     = 8000
      protocol    = "tcp"
      cidr_blocks = var.allowed_console_cidrs
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, { Name = "${var.project}-console-sg" })
}
