output "vpc_id" { value = aws_vpc.main.id }
output "public_subnet_id" { value = aws_subnet.public.id }
output "worker_security_group_id" { value = aws_security_group.worker.id }
output "console_security_group_id" { value = aws_security_group.console.id }
