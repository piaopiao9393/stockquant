"""测试默认使用AKShare作为数据源"""
import sys
sys.path.insert(0, 'src')

print("=" * 60)
print("测试默认数据源（应优先使用 AKShare）")
print("=" * 60)

from stockquant.data import create_data_fetcher, UnifiedDataFetcher

# 测试1：自动选择（应该使用AKShare）
print("\n[1] 测试自动选择数据源...")
fetcher = create_data_fetcher()
print(f"当前数据源: {fetcher.source_type.value}")
assert fetcher.source_type.value == 'akshare', "默认应该是AKShare"
print("[OK] 默认使用 AKShare")

# 测试2：明确指定AKShare
print("\n[2] 测试明确指定AKShare...")
fetcher2 = create_data_fetcher(prefer_source='akshare')
assert fetcher2.source_type.value == 'akshare'
print("[OK] 可以指定使用 AKShare")

# 测试3：获取数据
print("\n[3] 测试获取数据...")
df = fetcher.get_daily_prices('000001.SZ', start_date='20241201', end_date='20241210')
if not df.empty:
    print(f"[OK] 成功获取数据: {len(df)} 条")
    print(f"\n样本数据:")
    print(df[['trade_date', 'open', 'close']].head(3).to_string(index=False))
else:
    print("[X] 获取数据失败")

print("\n" + "=" * 60)
print("配置测试通过！AKShare 已设为默认数据源")
print("=" * 60)
