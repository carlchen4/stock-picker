#!/usr/bin/env python3
"""
🔍 TSX 量化选股系统 — 首次运行检查清单

用途：在第一次运行 python picker.py 之前，运行这个脚本验证所有依赖和配置

用法：
    python check_setup.py

输出：
    ✅ 绿色 = 已就绪
    ⚠️  黄色 = 可选（有备用方案）
    ❌ 红色 = 缺失，必须修复
"""

import sys
import subprocess
import os
from pathlib import Path

# 颜色输出
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

def check_python_version():
    """检查 Python 版本"""
    print(f"\n{BOLD}1️⃣  Python 版本{RESET}")
    v = sys.version_info
    print(f"   当前版本：Python {v.major}.{v.minor}.{v.micro}")
    if v.major == 3 and v.minor >= 10:
        print(f"   {GREEN}✅ 满足要求（>= 3.10）{RESET}")
        return True
    else:
        print(f"   {RED}❌ 版本过低（需要 >= 3.10）{RESET}")
        print(f"   {YELLOW}→ 升级：brew install python@3.12{RESET}")
        return False

def check_packages():
    """检查必要的 Python 包"""
    print(f"\n{BOLD}2️⃣  依赖包检查{RESET}")
    
    required = {
        "pandas": "数据处理",
        "numpy": "数值计算",
        "xgboost": "机器学习",
        "lightgbm": "梯度提升",
        "torch": "神经网络",
        "scikit-learn": "预处理",
        "yfinance": "股价数据",
        "requests": "API 请求",
        "cvxpy": "凸优化",
        "reportlab": "PDF 生成",
    }
    
    optional = {
        "ta-lib": "技术指标（不装也行）",
    }
    
    all_ok = True
    
    for pkg, desc in required.items():
        try:
            __import__(pkg)
            print(f"   {GREEN}✅ {pkg:15} {RESET} → {desc}")
        except ImportError:
            print(f"   {RED}❌ {pkg:15} {RESET} → {desc} (缺失)")
            all_ok = False
    
    print()
    for pkg, desc in optional.items():
        try:
            __import__(pkg)
            print(f"   {GREEN}✅ {pkg:15} {RESET} → {desc}")
        except ImportError:
            print(f"   {YELLOW}⚠️  {pkg:15} {RESET} → {desc} (可选)")
    
    if not all_ok:
        print(f"\n   {YELLOW}修复：pip install -r requirements.txt{RESET}")
    
    return all_ok

def check_directories():
    """检查项目目录结构"""
    print(f"\n{BOLD}3️⃣  目录结构{RESET}")
    
    base = Path("/Users/carlchenn/Documents/stock-picker")
    files_to_check = {
        "picker.py": "核心系统",
        "README.md": "项目文档",
        "OPERATIONS_MANUAL.md": "运维手册",
        "QUICK_REFERENCE.txt": "快速参考卡",
    }
    
    all_ok = True
    for fname, desc in files_to_check.items():
        fpath = base / fname
        if fpath.exists():
            size = fpath.stat().st_size / 1024  # KB
            print(f"   {GREEN}✅ {fname:30} {RESET} ({size:.0f} KB) → {desc}")
        else:
            print(f"   {RED}❌ {fname:30} {RESET} → {desc} (缺失)")
            all_ok = False
    
    # 检查缓存目录
    cache_dir = base / "simfin_data"
    if cache_dir.exists():
        print(f"   {GREEN}✅ simfin_data/{RESET:20} (缓存目录已存在)")
    else:
        print(f"   {YELLOW}⚠️  simfin_data/{RESET:20} (首次运行时自动创建)")
    
    return all_ok

def check_configuration():
    """检查 picker.py 的关键配置"""
    print(f"\n{BOLD}4️⃣  关键配置{RESET}")
    
    picker_path = Path("/Users/carlchenn/Documents/stock-picker/picker.py")
    
    if not picker_path.exists():
        print(f"   {RED}❌ 找不到 picker.py{RESET}")
        return False
    
    content = picker_path.read_text(encoding="utf-8")
    
    checks = {
        "MODE = \"pick\"": "运行模式",
        "MY_CURRENT_PORTFOLIO": "当前持仓字典",
        "EMAIL_CONFIG": "邮件配置",
        "BT_TX_COST": "交易成本",
        "XIU.TO": "基准 ETF（TSX 60）",
    }
    
    all_ok = True
    for pattern, desc in checks.items():
        if pattern in content:
            print(f"   {GREEN}✅ {desc:20} {RESET} (已配置)")
        else:
            print(f"   {RED}❌ {desc:20} {RESET} (配置缺失)")
            all_ok = False
    
    # 检查 MY_CURRENT_PORTFOLIO 是否有实际数据
    if "MY_CURRENT_PORTFOLIO = {" in content:
        # 简单检查：看看是否有实际的股票代码
        if '"' in content.split("MY_CURRENT_PORTFOLIO = {")[1][:200]:
            print(f"   {GREEN}✅ {'MY_CURRENT_PORTFOLIO 数据':20} {RESET} (已填入)")
        else:
            print(f"   {YELLOW}⚠️  {'MY_CURRENT_PORTFOLIO 数据':20} {RESET} (为空或示例，需更新)")
    
    return all_ok

def check_network():
    """检查网络连接"""
    print(f"\n{BOLD}5️⃣  网络服务{RESET}")
    
    services = {
        "yfinance": "https://finance.yahoo.com",
        "FMP API": "https://financialmodelingprep.com",
        "Gmail SMTP": "smtp.gmail.com:587",
    }
    
    for service, endpoint in services.items():
        try:
            if "smtp" in endpoint.lower():
                import smtplib
                server = smtplib.SMTP(endpoint.split(":")[0], int(endpoint.split(":")[1]), timeout=2)
                server.quit()
                print(f"   {GREEN}✅ {service:20} {RESET} (可连接)")
            else:
                import requests
                r = requests.head(endpoint, timeout=3)
                print(f"   {GREEN}✅ {service:20} {RESET} (可连接)")
        except Exception as e:
            print(f"   {YELLOW}⚠️  {service:20} {RESET} (暂时不可用，但会重试)")

def check_disk_space():
    """检查磁盘空间"""
    print(f"\n{BOLD}6️⃣  磁盘空间{RESET}")
    
    import shutil
    stat = shutil.disk_usage("/Users/carlchenn/Documents")
    free_gb = stat.free / (1024**3)
    
    print(f"   可用空间：{free_gb:.1f} GB")
    if free_gb > 1.0:
        print(f"   {GREEN}✅ 充足（需要 > 1 GB）{RESET}")
        return True
    else:
        print(f"   {RED}❌ 空间不足{RESET}")
        return False

def main():
    """主检查流程"""
    print(f"\n{BOLD}{'='*70}")
    print(f"  🔍 TSX 量化选股系统 — 首次运行检查清单")
    print(f"  {'='*70}{RESET}\n")
    
    results = {
        "Python 版本": check_python_version(),
        "依赖包": check_packages(),
        "目录结构": check_directories(),
        "配置项": check_configuration(),
        "磁盘空间": check_disk_space(),
    }
    
    check_network()  # 网络检查结果不影响总体判断
    
    # 总结
    print(f"\n{BOLD}{'='*70}")
    print(f"  检查结果总结")
    print(f"  {'='*70}{RESET}\n")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for item, status in results.items():
        symbol = "✅" if status else "❌"
        print(f"   {symbol} {item}")
    
    print(f"\n   总体：{passed}/{total} 通过\n")
    
    if passed == total:
        print(f"   {GREEN}{BOLD}🎉 所有检查通过！{RESET}")
        print(f"   {GREEN}你可以运行：python picker.py{RESET}\n")
        return 0
    else:
        print(f"   {RED}{BOLD}⚠️  有 {total-passed} 项检查失败，请修复后重试{RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
