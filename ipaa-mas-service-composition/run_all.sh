#!/usr/bin/env bash
# Linux / macOS equivalent of run_all.bat
set -u
cd "$(dirname "$0")"
PY=${PY:-python3}
$PY -m pip install -r requirements.txt
$PY download_data.py || true
mkdir -p results
: > results/run_log.txt
for E in e1_e2_benchmark e3_exact e4_structured e5_sensitivity e6_largescale e7_dynamic e8_negotiation; do
  echo "================= $E =================" | tee -a results/run_log.txt
  $PY $E.py >> results/run_log.txt 2>&1 || echo "[WARN] $E reported an error"
done
$PY summarize.py > results/summary.txt 2>&1
$PY -c "import platform,os;open('results/machine.txt','w').write(platform.platform()+'\n'+platform.processor()+'\ncpus='+str(os.cpu_count())+'\npython='+platform.python_version()+'\n')"
rm -f results.zip && (cd results && zip -qr ../results.zip .)
echo "Done. Please attach results.zip to the chat."
