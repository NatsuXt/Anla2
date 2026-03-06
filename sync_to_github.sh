#!/bin/bash
# =====================================================================
#  Project Anla — Git 清理与同步脚本
#  在项目根目录 (Anla/) 下执行
# =====================================================================
#
#  执行前请确认:
#    1. 当前在 Anla 项目根目录
#    2. 所有本地修改已保存
#    3. .gitignore, requirements.txt, README.md 已放到正确位置
#
#  此脚本做三件事:
#    [1] 从 git 追踪中移除可重新生成的二进制文件 (不删除本地文件)
#    [2] 添加新文件和修改后的文件
#    [3] 提交并推送
# =====================================================================

set -e  # 遇到错误立即退出

echo "=========================================="
echo "  Step 1: 移除可重新生成的训练截图"
echo "=========================================="

# 从 git 追踪中移除 (--cached 表示不删除本地文件)
# 这些是逐 epoch 的训练过程截图, 可通过重新运行实验生成
git rm --cached -r Logs/manifold_vis/ 2>/dev/null || echo "  manifold_vis/ 已清理或不存在"
git rm --cached -r Logs/manifold_vis_permuted/ 2>/dev/null || echo "  manifold_vis_permuted/ 已清理或不存在"
git rm --cached -r Logs/recursive_k5_vis/ 2>/dev/null || echo "  recursive_k5_vis/ 已清理或不存在"
git rm --cached -r Logs/ring_transformer_vis/ 2>/dev/null || echo "  ring_transformer_vis/ 已清理或不存在"

echo ""
echo "=========================================="
echo "  Step 2: 暂存所有变更"
echo "=========================================="

# 添加新/修改的文件
git add .gitignore
git add requirements.txt
git add README.md

# v4.4 修改的源文件
git add losses/boltzmann_elegant.py
git add experiments/capacity/capacity_pressure_test_v4.py
git add layers/activation.py
git add visualization/visualize_config_B.py

# v4.4 新实验结果 (如果已运行并生成)
git add Logs/config_B_analysis/ 2>/dev/null || echo "  config_B_analysis/ 无新变更"

echo ""
echo "=========================================="
echo "  Step 3: 确认变更"
echo "=========================================="

echo ""
echo "将要提交的变更:"
git status
echo ""

# 统计
echo "追踪文件数量 (提交后):"
git ls-files | wc -l
echo ""
echo "其中二进制文件:"
git ls-files | grep -E '\.(png|npz|pdf|jpg|jpeg)$' | wc -l
echo ""

read -p "确认提交? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "=========================================="
    echo "  Step 4: 提交并推送"
    echo "=========================================="
    
    git commit -m "v4.4: τ排除target修复 + force_b排斥力 + 双轨分析

核心修改:
- [losses] τ = std(E_competitors): 排除target能量, 消除自反馈崩溃
  τ波动从10→38缩小到9.8~11.4, p_target从0.17恢复到0.97
- [losses] force_b升级为Term1 Boltzmann力(含排斥), Self-Boltzmann p_target 0.62→0.97
- [train] Early stopping改为accuracy AND loss双重判据
- [train] 双轨分析: best checkpoint + final model同时评估, 退化幅度可量化
- [activation] PhaseTwist m clamp防止负值边界不连续

Config B (V=256, D=64) 结果:
- Test Acc: 75.9%±6.7% → 88.9%±2.7% (+13%, 方差↓)
- neg%: 24.8% → 0.0%
- 训练全程稳定, best≈final (退化+0.52%)

清理:
- 移除16.5MB可重新生成的逐epoch训练截图
- 新增 .gitignore, requirements.txt, README.md"

    git push origin main
    
    echo ""
    echo "=========================================="
    echo "  完成!"
    echo "=========================================="
else
    echo "已取消。你可以手动执行 git commit 和 git push。"
fi
