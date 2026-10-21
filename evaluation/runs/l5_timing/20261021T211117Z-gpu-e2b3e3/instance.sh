set -euo pipefail
# L5 quiet-host timings (run by spinup.sh on a fresh g4dn.xlarge; nothing else runs on it).
REG=836505524290.dkr.ecr.eu-north-1.amazonaws.com
IMG=$REG/ariadne/ariadne-core:l5-7f39b6f98cc0
PAY=s3://ariadne-836505524290-eun1/runs/l5-payload-20261021T210611Z/payload.tar.gz
R=$ARIADNE_RESULTS
command -v docker >/dev/null || { apt-get update -qq && apt-get install -y -qq docker.io; }
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin $REG
docker pull -q $IMG
docker image inspect $IMG --format '{{.Id}}' > $R/image_id.txt
mkdir -p /opt/l5 && aws s3 cp $PAY - | tar xz -C /opt/l5
(cd /opt/l5/p && sha256sum -c --quiet SHA256SUMS) && echo "payload sha256 OK" | tee $R/payload_check.txt
cp /opt/l5/p/SHA256SUMS $R/
lscpu > $R/lscpu.txt; nproc > $R/nproc.txt; free -m > $R/free.txt
mkdir -p /opt/l5/out && chmod 777 /opt/l5/out
for t in 1 4; do
  { date -u +%FT%TZ; cat /proc/loadavg; } > $R/load_before_t$t.txt
  docker run --rm --network none -v /opt/l5/p:/p:ro -v /opt/l5/out:/out $IMG /p/ariadne_dnn_timing \
    --xfeat /p/xf --ep /p/ep --ep-sim /p/eps --al /p/al --al-sim /p/als \
    --threads $t --reps 30 --warmup 3 --out /out/timing_t$t.csv 2>&1 | grep -v WARN | tee $R/timing_t$t.log
  { date -u +%FT%TZ; cat /proc/loadavg; } > $R/load_after_t$t.txt
done
cp /opt/l5/out/*.csv $R/
docker ps -a > $R/docker_ps.txt
