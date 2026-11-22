"""Usage: --plan JSON(list of pinned package entries) --output new-report.json."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.first_round_gate import check_packages

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--plan',required=True)
p.add_argument('--output',required=True)
args=p.parse_args()
entries=json.loads(Path(args.plan).read_text(encoding='utf-8'))
result=check_packages(entries)
with Path(args.output).open('x',encoding='utf-8') as handle:
    json.dump(result,handle,indent=2);handle.write('\n')
print(json.dumps(result))
