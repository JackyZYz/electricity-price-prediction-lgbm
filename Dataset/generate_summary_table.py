import pandas as pd
import glob
import os
from datetime import datetime

files = sorted(glob.glob('**/*.csv', recursive=True))
records = []
for f in files:
    df = pd.read_csv(f)
    n_rows, n_cols = df.shape
    cols = list(df.columns)
    date_cols = [c for c in cols if '时间' in c or '日期' in c or 'time' in c.lower() or 'date' in c.lower()]
    date_range = None
    for dc in date_cols:
        try:
            s = pd.to_datetime(df[dc], errors='coerce')
            valid = s.dropna()
            if len(valid) > 0:
                date_range = (valid.min().strftime('%Y-%m-%d'), valid.max().strftime('%Y-%m-%d'))
                break
        except Exception:
            continue
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    missing = df.isnull().sum().sum()
    folder = os.path.dirname(f).replace('\\', '/')
    name = os.path.basename(f)
    records.append({
        '文件夹': folder if folder else '根目录',
        '文件名': name,
        '行数': n_rows,
        '列数': n_cols,
        '数值列数': len(numeric_cols),
        '缺失值': int(missing),
        '起始日期': date_range[0] if date_range else None,
        '结束日期': date_range[1] if date_range else None,
    })

df_sum = pd.DataFrame(records)
df_sum.to_csv('数据集汇总表.csv', index=False, encoding='utf-8-sig')
print(df_sum.to_string(index=False))
