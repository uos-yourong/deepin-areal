# AReaL + osworld 训练架构文档

## 功能特性

- **Env Pool 单机多线程**：基于 libvirt/KVM 的并发 VM Pool，单机多 worker 轮询调度
- **Reward LLM Judge**：Reward 由 `vlm_judge.py` 实现
- **LoRA 补丁**：挂载 `fsdp_engine.py` 补丁解决 LoRA 包裹后找不到 `compute_3d_position` 的问题

---

## 1. 问题提出

**AReaL 没有支持 virsh 启动 Deepin 环境**

osworld没有支持virsh启动deepin,所以用os_env 启动vm(env) pool：用 `virsh` 启动 Deepin OS，在 VM 内部启动 `launch`、`screen` 等服务。

---

## 2. 最终架构

> 忽略宿主机的 AReaL v1.4，容器内使用 AReaL v2.0

环境分为两块：

| 环境 | 镜像/路径 | 用途 |
|------|----------|------|
| 训练+Rollout | `ghcr.io/areal-project/areal-runtime:v2.0.0-vllm` | vLLM rollout、FSDP actor、训练 |
| 环境交互 | `os_env` 文件夹（见 `os_env/`） | VM 管理、env_worker、Deepin 桌面环境 |

- **宿主机**：`os_env`
- **容器内**：`rollout` + `训练`

### 2.1 其他已尝试并否定的架构

| 方案 | 说明 | 否定原因 |
|------|------|----------|
| 宿主机 `env_os`，容器内 `训练` + `rollout` | — | 本机 CUDA 太老无法升级 |
| 容器内 `env_os` + `训练` + `rollout` | — | Torch、Qwen3、CUDA、AReaL 同时满足兼容太难 |

---

## 3. 调试的 Bug 与补丁

### 3.1 LoRA 包裹后没有 `compute_3d_position` 的问题

- **补丁文件**：`/AReaL/areal/engine/fsdp_engine.py`
- **挂载路径**：`/mnt/data/yr/code/rl_osworld/AReaL-2.0-running` 挂载到容器
- **其他挂载**：除了 `env_os`，`example/osworld` 也挂载到容器

### 3.2 并发的 VM Pool

#### 3.2.1 整体架构图

```
┌─────────────────────────── 宿主机 ───────────────────────────┐
│                                                              │
│  libvirt/KVM                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │
│  │ VM rl-1  │ │ VM rl-2  │ │ VM rl-3  │ │ VM rl-4  │         │
│  │ deepin   │ │ deepin   │ │ deepin   │ │ deepin   │         │
│  │ 6C/12G   │ │ 6C/12G   │ │ 6C/12G   │ │ 6C/12G   │         │
│  └────▲─────┘ └────▲─────┘ └────▲─────┘ └────▲─────┘         │
│       │            │            │            │                │
│  ┌────┴────┐  ┌────┴────┐  ┌────┴────┐  ┌────┴────┐          │
│  │worker:8300  │worker:8301  │worker:8302  │worker:8303      │
│  │env_worker│  │env_worker│  │env_worker│  │env_worker│      │
│  │ 进程A    │  │ 进程B    │  │ 进程C    │  │ 进程D    │       │
│  └────▲────┘  └────▲────┘  └────▲────┘  └────▲────┘          │
│       │            │            │            │                │
│   :8300         :8301        :8302        :8303               │
│       └────────────┴─────┬──────┴────────────┘                │
│                          │ HTTP（host network）                │
└──────────────────────────┼────────────────────────────────────┘
                           │
┌──────────────────────────┼────────────────────────────────────┐
│  deepin-train 容器       ▼                                    │
│  ┌─────────────────────────────────────────────┐             │
│  │ DeepinWorkflow（rollout worker 进程内）       │             │
│  │                                             │             │
│  │  arun_episode(task, n_trajs=4)              │             │
│  │    └─ asyncio.gather ← 4 条轨迹并发          │             │
│  │         ├─ traj0 → _build_env() → 取URL 8300 │             │
│  │         ├─ traj1 → _build_env() → 取URL 8301 │             │
│  │         ├─ traj2 → _build_env() → 取URL 8302 │             │
│  │         └─ traj3 → _build_env() → 取URL 8303 │             │
│  │                                             │             │
│  │  每个 = RemoteDesktopEnv（HTTP client）       │             │
│  │  reset → step×15 → close(=reset回快照)       │             │
│  └─────────────────────────────────────────────┘             │
└───────────────────────────────────────────────────────────────┘
```

#### 3.2.2 三层各自的职责

**VM 层（libvirt + deepin）**

- 每个 VM 是模板 `deepin-template-1010.qcow2` 的 **COW 克隆**（秒级创建，不占真实磁盘空间直到写入）
- 共享协调设施（都在 `os_env/state/`）：
  - `.libvirt_lck` 文件锁——多个 worker 同时创建 VM 时串行排队
  - `.libvirt_vms` 注册表——`deepin-vm-rl-N|状态`，避免重名
  - VNC 端口分配（7901 起递增）

**Worker 层（`env_worker.py`，一进程一 VM）**

- 每个 worker 启动时从池里领一个 VM（自己创建或复用空闲的），**独占**它
- FastAPI 服务 5 个端点：
  - `/ping` —— 健康检查
  - `/reset` —— 恢复快照 + 执行任务 setup
  - `/step` —— 执行 pyautogui 动作
  - `/observe` —— 截图
  - `/close` —— 关闭
- **关键设计**：`close_mode="reset"` 时训练侧的 `close` 调的是 `/reset` 而不是 `/close`——VM 回到快照、worker 进程活着，下一条轨迹来了直接复用，不用重新起 VM

**分配层（~15 行）**

- 没有「中央调度器」——分配逻辑就在 workflow 里，一个 `itertools.cycle(urls)` 加锁轮询：

```python
with self._env_url_lock:          # _build_env 被并发线程调用
    url = next(self._env_url_cycle)   # 8300→8301→8302→8303→8300...
return RemoteDesktopEnv(base_url=url, ...)
```

- 并发轨迹数 ≤ worker 数时，每条轨迹天然独占一个 VM；超出时退化为共享（和改造前一样，不劣化）

#### 3.2.3 一条轨迹的完整生命周期

```
traj 开始
  → _build_env()：从 cycle 领一个 worker URL
  → POST /reset：VM 恢复快照 + 执行任务初始 setup（下载文件/启动 Blender）
  → 循环 ≤15 步：截图(observe) → vLLM 生成动作 → POST /step 执行
  → 轨迹结束：close() → POST /reset（VM 回快照，worker 留守）
  → 下一条轨迹来时，这个 worker 又是干净状态
```

#### 3.2.4 为什么不用 os_env 自带的中央池（`start.py`）

`os_env` 里确实有一套更重的方案：中央网关（8081 端口）+ session 管理（`/session/start_batch`）+ `workers_pool.json`。它是为多节点 Ansible 部署设计的。

| 维度 | 轮询方案（现用） | `start.py` 中央池 |
|------|-----------------|-------------------|
| workflow 侧改动 | 15 行 | 客户端要重写成 session API |
| 组件数 | N 个 worker，无中心 | N worker + 1 网关 + 池状态文件 |
| 分配决策 | 静态轮询 | 动态调度、崩溃处理 |
| 适用场景 | 单机、固定并发 | 多机、动态扩缩 |

> 单机静态池用轮询就是最优解；将来如果要上多机，再切 `start.py` 也只是换 `_build_env` 的实现。

#### 3.2.5 运维边界

| 操作 | 说明 |
|------|------|
| `start_env_pool.sh N` | 幂等起池（ping 通的跳过），等全部就绪才返回 |
| `stop_env_pool.sh` | 全杀 + 提示清 VM |
| worker 挂了 | 它那条轨迹失败被丢弃（GRPO 的失败语义），池不会自动补——目前接受，以后可加健康检查/自动重启 |

### 3.3 Reward

Reward 由 `vlm_judge.py` 实现，新的流程在 `osworld/workflow/deepin_workflow.py` 中。

---

## 4. 启动流程与补丁加载

```
启动 env pool
    ↓
启动 Docker container（注意挂载）
    ↓
run lora test、env_worker test
    ↓
关闭 env pool
```

---

## 5. 训练 / Rollout / Env_OS 的最终架构

```
┌───────────────────── 宿主机 ─────────────────────┐
│                                                    │
│  env_worker.py (os_env) ←── libvirt → deepin VM    │
│       :8300 (HTTP)                                 │
│         ▲                                          │
│         │ HTTP (reset/step/observe/close)          │
│         │                                          │
│  ┌───────┴────────── Docker 容器 ──────────────┐   │
│  │  deepin-train (host network, GPU 4+5)       │   │
│  │                                              │   │
│  │  vLLM rollout (GPU 4)  ←  Qwen3-VL-8B       │   │
│  │       :port (AReal 管理)                     │   │
│  │                                              │   │
│  │  FSDP actor + ref (GPU 5)                    │   │
│  │  DeepinWorkflow                              │   │
│  │  mock VLMJudge                               │   │
│  └──────────────────────────────────────────────┘   │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## 附录

- AReaL 版本：v2.0（容器内）
- 模型：Qwen3-VL-8B
- GPU 分配：GPU 4（vLLM rollout）、GPU 5（FSDP actor + ref）
- 容器网络模式：`host network`

---

## 6. 创建 deepin-train 容器

### 6.1 Docker 启动命令

```bash
docker run -d --name deepin-train   --gpus '"device=4,5"'   --privileged --network host --ipc=host   --dns 8.8.8.8 --dns 114.114.114.114   -v /var/run/libvirt/libvirt-sock:/var/run/libvirt/libvirt-sock   -v /var/lib/libvirt:/var/lib/libvirt   -v /mnt/data/yr/GUI_env/vm_image:/mnt/data/yr/GUI_env/vm_image   -v /mnt/data/yr/code/rl_osworld/os_env:/os_env   -v /mnt/data/yr/code/rl_osworld/任务数据:/任务数据   -v /mnt/data/yr/download/Qwen/Qwen3-VL-8B-Instruct:/model   -v /mnt/data/yr/code/rl_osworld/AReaL-2.0-running:/AReaL   -v /mnt/data/yr/code/rl_osworld/AReaL-feat-osworld-grpo-example/examples/osworld:/AReaL/examples/osworld   -v /mnt/data/yr/code/rl_osworld/tmp1:/tmp1   -e PYTHONUNBUFFERED=1   -w /AReaL   --entrypoint sleep   ghcr.io/areal-project/areal-runtime:v2.0.0-vllm   infinity
```

### 6.2 挂载说明

| 宿主机路径 | 容器内路径 | 说明 |
|-----------|-----------|------|
| `/var/run/libvirt/libvirt-sock` | `/var/run/libvirt/libvirt-sock` | libvirt socket |
| `/var/lib/libvirt` | `/var/lib/libvirt` | libvirt 数据 |
| `/mnt/data/yr/GUI_env/vm_image` | `/mnt/data/yr/GUI_env/vm_image` | VM 镜像 |
| `/mnt/data/yr/code/rl_osworld/os_env` | `/os_env` | os_env 代码 |
| `/mnt/data/yr/code/rl_osworld/任务数据` | `/任务数据` | 任务数据 |
| `/mnt/data/yr/download/Qwen/Qwen3-VL-8B-Instruct` | `/model` | Qwen3 模型 |
| `/mnt/data/yr/code/rl_osworld/AReaL-2.0-running` | `/AReaL` | AReaL 2.0 核心代码（含 LoRA 补丁） |
| `/mnt/data/yr/code/rl_osworld/AReaL-feat-osworld-grpo-example/examples/osworld` | `/AReaL/examples/osworld` | workflow/配置（优先级更高） |
| `/mnt/data/yr/code/rl_osworld/tmp1` | `/tmp1` | 临时目录 |

### 6.3 注意事项

- `/AReaL` 挂载的是 `AReaL-2.0-running`（从旧容器拷出、git 固化的 areal 2.0，已含 LoRA 补丁：`fsdp_engine.py` 的 PEFT 解包）。以后改 areal 核心代码就改这个目录，改完 `git commit` 留痕，容器删了重建补丁也不丢。
- `examples/osworld` 挂载在更深层路径，优先级更高——改 workflow/配置还是改 `AReaL-feat-osworld-grpo-example/examples/osworld/`
- 容器内首次用 git 需先执行：
  ```bash
  docker exec deepin-train git config --global --add safe.directory /AReaL
  ```

### 6.4 以后 debug 的工作流

| 修改目标 | 编辑位置 |
|---------|---------|
| AReaL 核心（fsdp_engine 等） | `AReaL-2.0-running/` → `git commit` |
| workflow/配置 | `AReaL-feat-osworld-grpo-example/examples/osworld/` |
| 容器重建 | 补丁都在挂载里，不会丢 |

---

## 7. 执行顺序

### 7.1 第一步：宿主机起 env_worker（VM 环境，耗时 ~30s）

```bash
### 不用这个用8 是可以的
cd /mnt/data/yr/code/rl_osworld/os_env && MANAGE_UFW_FOR_WORKER_PORT=0 setsid nohup /mnt/data/yr/code/rl_osworld/myenv_areal/bin/python desktop_env/env_worker.py   --provider libvirt --os-type deepin --headless   --port 8300 --host 127.0.0.1   --cache-dir /mnt/data/yr/code/rl_osworld/tmp1/worker_cache   </dev/null >> /mnt/data/yr/code/rl_osworld/tmp1/env_worker.log 2>&1 &
```

**为什么用 `setsid` 和 `</dev/null`：**

worker 内部会调 `sudo virsh`，sudo 默认 `use_pty` 会在你的终端上做 I/O 中继；而后台进程（`&`）一碰控制终端就被内核挂起，整个 worker 无声冻结。`setsid` 让它脱离终端会话就没有这个问题了。

**配套规矩（都是踩过的坑）：**

1. **绝不要 Ctrl+Z**——挂起的进程不释放 libvirt 锁，会卡死后续所有实例
2. **启动前确认旧实例已清干净**：
   ```bash
   ps aux | grep env_worker | grep -v grep   # 应为空
   ss -tlnp | grep 8300                        # 应为空
   ```
3. **起完耐心等 ~60 秒**，看到 `Env Worker started` + `Uvicorn running` 才算就绪，再 `curl http://127.0.0.1:8300/ping` 确认
4. **如果要杀掉重来**：
   ```bash
   pkill -9 -f env_worker.py
   # 残留 VM 清理
   sudo virsh destroy deepin-vm-rl-1
   sudo virsh undefine deepin-vm-rl-1
   ```

### 7.2 第二步：确认 deepin-train 容器在跑

```bash
docker ps | grep deepin-train
```

如果不在，用 6.1 节的 `docker run` 命令创建。

### 7.3 第三步：容器内跑训练（smoke 配置，耗时 ~10 分钟）

```bash
docker exec deepin-train bash -c '
cd /AReaL && VLM_API_BASE=mock VLM_MODEL=mock CUDA_VISIBLE_DEVICES=4,5 PYTHONUNBUFFERED=1 python -m examples.osworld.train_deepin --config examples/osworld/config_deepin_vllm.yaml trial_name=smoke-lora6 rollout.max_concurrent_rollouts=1 n_trajs=1 max_steps=2 train_dataset.batch_size=1 total_train_epochs=1
'
```

---

### 7.4 第三步（变体）：使用真实 VLM API（Kimi）

```bash
docker exec deepin-train bash -c '
cd /AReaL && VLM_API_BASE=https://101.89.57.42:5911 VLM_API_KEY=xxx VLM_MODEL=kimi CUDA_VISIBLE_DEVICES=4,5 PYTHONUNBUFFERED=1 python -m examples.osworld.train_deepin --config examples/osworld/config_deepin_vllm.yaml trial_name=real-reward-smoke rollout.max_concurrent_rollouts=1 n_trajs=4 max_steps=2 train_dataset.batch_size=1 total_train_epochs=1
'
```

## 8. Env Pool 的启动和关闭

```bash
# 起池（训练前）
/mnt/data/yr/code/rl_osworld/start_env_pool.sh 4 ### 4 vm

# 跑训练（n_trajs 可放心设 4，甚至 max_concurrent_rollouts=2 配 8 个 worker）
docker exec deepin-train bash -c '... trial_name=xxx n_trajs=4 ...'

# 收工停池（可选，不停也行，worker 空闲不占 GPU）
/mnt/data/yr/code/rl_osworld/stop_env_pool.sh
```
