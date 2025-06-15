#!/usr/bin/env python3
import os
import pandas as pd
from datasets import load_dataset, concatenate_datasets
import random

# Load the two parts of AIME2025
ds1 = load_dataset("opencompass/AIME2025", "AIME2025-I")['test']
ds2 = load_dataset("opencompass/AIME2025", "AIME2025-II")['test']
ds = concatenate_datasets([ds1, ds2])

# Loop over your num_tokens variants
for num_tokens in [512, 1024, 2048, 3600, -512, -1024, -2048, -3600, -1]:
    all_data = []
    for i in range(len(ds)):
        question = ds[i]['question'].strip()
        prompt = question + "\n\nLet's think step by step and output the final answer within \\boxed{}." \
                 + (f" Think for {num_tokens} tokens." if num_tokens != -1 else "")
        all_data.append({
            "data_source": "aime2025",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": ds[i]['answer'],
                "num_tokens": num_tokens
            },
            "extra_info": {'split': 'test', 'index': i}
        })
    df = pd.DataFrame(all_data)

    # Determine output directory & path
    if num_tokens == -1:
        out_dir = os.path.expanduser('~/deepscaler/data')
    else:
        # NOTE: preserve your original naming convention here:
        if num_tokens < 0:
            out_dir = os.path.expanduser(f'~/deepscaler/data9_{num_tokens}')
        else:
            out_dir = os.path.expanduser(f'~/deepscaler/data_{num_tokens}')

    # 1) make sure the directory exists
    os.makedirs(out_dir, exist_ok=True)
    # 2) build full path
    out_path = os.path.join(out_dir, 'aime2025.parquet')
    # 3) write parquet
    df.to_parquet(out_path, index=False)

    print(f"Wrote {len(df)} examples to {out_path}")

    
