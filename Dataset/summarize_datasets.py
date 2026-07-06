import pandas as pd
import os
import glob
from datetime import datetime

files = sorted(glob.glob('**/*.csv', recursive=True))

results = []
for f in files:
    try:
        df = pd.read_csv(f)
        n_rows, n_cols = df.shape
        cols = list(df.columns)

        # date column detection
        date_cols = [c for c in cols if '时间' in c or '日期' in c or 'time' in c.lower() or 'date' in c.lower()]
        date_range = None
        for dc in date_cols:
            try:
                s = pd.to_datetime(df[dc], errors='coerce')
                valid = s.dropna()
                if len(valid) > 0:
                    date_range = (valid.min().strftime('%Y-%m-%d %H:%M:%S'), valid.max().strftime('%Y-%m-%d %H:%M:%S'))
                    break
            except Exception:
                continue

        # numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        num_numeric = len(numeric_cols)

        # missing values
        missing = df.isnull().sum().sum()

        # sample values from first numeric column
        sample = ''
        if numeric_cols:
            col = numeric_cols[0]
            vals = df[col].dropna().head(3).tolist()
            sample = ', '.join([f'{v:.4f}' if isinstance(v, float) else str(v) for v in vals])

        results.append({
            '文件路径': f,
            '行数': n_rows,
            '列数': n_cols,
            '列名': cols,
            '数值列数': num_numeric,
            '缺失值总数': int(missing),
            '时间范围': date_range,
            '首列数值样例': sample
        })
    except Exception as e:
        results.append({
            '文件路径': f,
            '错误': str(e)
        })

# Print summary
for r in results:
    print('='*80)
    for k, v in r.items():
        print(f'{k}: {v}')

# Also write to a markdown report
with open('数据集信息说明.md', 'w', encoding='utf-8') as out:
    out.write('# 数据集信息说明\n\n')
    out.write(f'生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
    out.write(f'文件总数：{len(results)}\n\n')

    for i, r in enumerate(results, 1):
        out.write(f'## {i}. {r["文件路径"]}\n\n')
        if '错误' in r:
            out.write(f'- 读取错误：{r["错误"]}\n\n')
            continue
        out.write(f'- 行数：{r["行数"]}\n')
        out.write(f'- 列数：{r["列数"]}\n')
        out.write(f'- 数值列数：{r["数值列数"]}\n')
        out.write(f'- 缺失值总数：{r["缺失值总数"]}\n')
        out.write(f'- 时间范围：{r["时间范围"] if r["时间范围"] else "未检测到时间列"}\n')
        out.write(f'- 列名：{", ".join(r["列名"])}\n')
        out.write(f'- 首列数值样例：{r["首列数值样例"]}\n\n')
