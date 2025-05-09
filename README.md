df['extracted'] = df['col'].str.extract(r'(?:L\s+)?(.+?)(?:_\d{6}|\(合計値\))?$')
