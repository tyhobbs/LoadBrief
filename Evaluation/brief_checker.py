import json
from collections import Counter
c = Counter()
for line in open('formatted/sft_test.jsonl'):
    c[json.loads(line)['scenario_type']] += 1
for s, n in sorted(c.items(), key=lambda x: x[1]):
    print(f'{n:5d}  {s}')