"""简化的AKShare测试 - 使用单一股票测试避免连接问题"""
import sys
sys.path.insert(0, 'src')

print("=" * 60)
print("测试 AKShare 免费数据源")
print("=" * 60)

try:
    from stockquant.data.akshare_fetcher import AKShareFetcher

    # 创建获取器（禁用代理以解决连接问题）
    print("\n[1] 初始化 AKShareFetcher...")
    fetcher = AKShareFetcher(disable_proxy=True)
    print("[OK] AKShare 初始化成功")

    # 测试获取单个股票日线数据（不获取全量股票列表，避免大数据量请求）
    print("\n[2] 获取单个股票日线数据 (000001.SZ 平安银行)...")
    df = fetcher.get_daily_prices('000001.SZ', start_date='20241201', end_date='20241220')

    if not df.empty:
        print(f"[OK] 获取到 {len(df)} 条日线数据")
        print("\n数据样本:")
        print(df[['trade_date', 'open', 'close', 'high', 'low', 'vol']].head(3).to_string(index=False))
    else:
        print("[X] 未获取到日线数据，可能是网络连接问题")
        print("建议: 检查网络连接，或稍后再试")

    # 测试交易日历
    print("\n[3] 获取交易日历...")
    calendar = fetcher.get_trade_calendar('20241201', '20241231')
    if not calendar.empty:
        print(f"[OK] 12月交易日数量: {len(calendar)}")
    else:
        print("[X] 未获取到交易日历")

    # 测试最新交易日
    print("\n[4] 获取最新交易日...")
    latest = fetcher.get_latest_trade_date()
    print(f"[OK] 最新交易日: {latest}")

    print("\n" + "=" * 60)
    print("AKShare 基础测试完成")
    print("=" * 60)
    print("\n说明:")
    print("- AKShare 是免费数据源，无需API Token")
    print("- 如果遇到连接问题，可能是网络环境或代理设置导致")
    print("- 可以替换 Tushare 作为项目的数据源")

except Exception as e:
    print(f"\n[X] 测试失败: {e}")
    print("\n可能的原因:")
    print("1. AKShare 未安装: pip install akshare")
    print("2. 网络连接问题")
    print("3. 代理设置问题")
    import traceback
    traceback.print_exc()
