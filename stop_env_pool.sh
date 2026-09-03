#!/bin/bash
# 停止全部 env_worker 并提示清理 VM
# 用法: ./stop_env_pool.sh [--clean]   --clean 时附带杀掉进程的命令提示

set -u
echo "=== 停止所有 env_worker ==="
pkill -9 -f "env_worker.py" && echo "已发送 kill 信号" || echo "没有运行中的 env_worker"
sleep 2

LEFT=$(pgrep -f "env_worker.py" | wc -l)
if [ "$LEFT" -gt 0 ]; then
    echo "警告：仍有 $LEFT 个进程残留："
    pgrep -af "env_worker.py"
    exit 1
fi
echo "✓ 进程已清空（端口 8300+ 应全部释放）"

echo ""
echo "如需彻底清理 VM（腾内存/磁盘），执行："
echo "  sudo virsh list --all | grep deepin-vm-rl"
echo "  sudo virsh destroy deepin-vm-rl-N        # 逐个停（在 running 时）"
echo "  sudo virsh undefine deepin-vm-rl-N --nvram   # 逐个删除定义"
echo "  rm -f /mnt/data/yr/GUI_env/vm_image/rl/deepin-vm-rl-*.qcow2   # 删 COW 磁盘"
