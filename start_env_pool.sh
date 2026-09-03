#!/bin/bash
# 启动 env_worker 池（N 个 worker，每个独占一个 deepin VM）
# 用法: ./start_env_pool.sh [N]   默认 N=4，端口 8300..8300+N-1
#
# 注意事项（踩过的坑）：
# - 必须 setsid + </dev/null：worker 内部调 sudo virsh，sudo use_pty 会在
#   控制终端做 I/O 中继，后台进程碰终端会被内核挂起（T 状态）
# - VM 创建在 libvirt 注册锁上串行，4 个 worker 全就绪约需 3~5 分钟
# - 绝不要 Ctrl+Z 挂起 worker（不释放锁，卡死后续实例）

set -u
N=${1:-4}
BASE_PORT=8300
ROOT=/mnt/data/yr/code/rl_osworld
OS_ENV=$ROOT/os_env
PY=$ROOT/myenv_areal/bin/python
LOGDIR=$ROOT/tmp1

ping_ok() { curl -s -m 3 "http://127.0.0.1:$1/ping" 2>/dev/null | grep -q '"status"'; }

echo "=== env_worker 池启动：N=$N，端口 ${BASE_PORT}..$((BASE_PORT+N-1)) ==="

# 第一阶段：逐个启动（已活的跳过）
for ((i=0; i<N; i++)); do
    PORT=$((BASE_PORT+i))
    LOG=$LOGDIR/env_worker_${PORT}.log
    if ping_ok $PORT; then
        echo "[${PORT}] 已在运行，跳过"
        continue
    fi
    # 端口被占但 ping 不通 → 有残骸，先杀掉
    if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
        echo "[${PORT}] 端口被占用但 ping 不通，清理残骸..."
        pkill -9 -f "env_worker.py.*--port ${PORT}" 2>/dev/null
        sleep 2
    fi
    echo "[${PORT}] 启动中... 日志: $LOG"
    cd $OS_ENV && \
    MANAGE_UFW_FOR_WORKER_PORT=0 \
    setsid nohup $PY desktop_env/env_worker.py \
      --provider libvirt \
      --os-type deepin \
      --headless \
      --port $PORT \
      --host 127.0.0.1 \
      --cache-dir $ROOT/tmp1/worker_cache \
      </dev/null >> $LOG 2>&1 &
    # 间隔几秒，让 libvirt 锁排队别太挤
    sleep 3
done

# 第二阶段：等全部就绪（VM 创建串行，最长 ~15 分钟）
echo ""
echo "=== 等待全部就绪（VM 串行创建，4 个约 3~5 分钟，请耐心）==="
TIMEOUT=900
ELAPSED=0
while true; do
    READY=0
    for ((i=0; i<N; i++)); do
        ping_ok $((BASE_PORT+i)) && READY=$((READY+1))
    done
    echo "[$(date +%H:%M:%S)] 就绪 $READY/$N  (已等 ${ELAPSED}s)"
    [ $READY -eq $N ] && break
    [ $ELAPSED -ge $TIMEOUT ] && { echo "超时！未就绪的 worker 看各自日志排查"; exit 1; }
    sleep 20
    ELAPSED=$((ELAPSED+20))
done

echo ""
echo "=== 池就绪 ==="
for ((i=0; i<N; i++)); do
    PORT=$((BASE_PORT+i))
    echo "[${PORT}] $(curl -s -m 3 http://127.0.0.1:${PORT}/ping)"
done
echo ""
echo "VM 列表检查（需要 sudo）: sudo virsh list --all | grep deepin-vm-rl"
