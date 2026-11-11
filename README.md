# vla_infra

Flexiv Rizon 4s 的 VLA 基础设施：本地工控机连接机械臂、GN01 与相机，远端 GPU 运行策略，本地负责采集、校验、安全门禁和执行。

当前版本已经通过 fake/injected-SDK 自动化测试，并提供 Flexiv **只读联机探针**。它尚未通过真实设备现场验收，因此：

> 当前只允许做 10 秒只读联机。禁止 Enable、清故障、切模式、Tool.Switch、夹爪写入和任何运动指令。这里的代码不是经过认证的机器人安全系统。

## 当前状态

| 能力 | 状态 |
|---|---|
| fake 数据采集、回放、shadow loop | 可用 |
| OpenPI 远程推理协议与 fake 闭环 | 可用，需现场网络验收 |
| Flexiv RDK 单臂只读探针 | 可做首次 10 秒现场验证 |
| RealSense/Flexiv 只读 adapter | 已有代码，未完成整站 soak |
| GN01 只读宽度采集 | 未完成 |
| 完整真实 VLA 数据采集 | 禁止 |
| 实机 Enable/运动/VLA rollout | 禁止 |

最新实机门禁见 [Field 01 验收说明](exchange/leader/2026-1111-1139/LEADER_M5_FIELD01_READONLY_GATE.md)。

## 工控机要求

- Ubuntu 22.04 或更新版本，x86_64；
- Python 3.10（与本项目及 Flexiv RDK 1.9.3 基线一致）；
- 工控机通过独立有线网卡连接 Flexiv 控制柜；
- 现场操作员、可触达急停、清空的工作区；
- 数据盘必须是最终用于录制的数据盘；
- Flexiv RDK Python 1.9.3。RealSense 采集阶段还需要 `pyrealsense2`；
- ROS2 **不是本项目的必要依赖**。不要同时启动 ROS2 writer、Flexiv 示例程序或其他 RDK 控制进程。

## 从 GitHub 安装

将下面的 URL 替换成实际仓库地址：

```bash
git clone https://github.com/<OWNER>/<REPOSITORY>.git
cd <REPOSITORY>

./scripts/bootstrap_ubuntu.sh
source .venv/bin/activate
```

脚本只创建 `.venv` 并安装本仓库的基础 Python 包，不安装系统服务，也不接触硬件。

安装 Flexiv RDK：

```bash
python -m pip install 'flexivrdk==1.9.3' spdlog
python -c "import flexivrdk; print(getattr(flexivrdk, '__version__', 'unknown'))"
```

如果站点使用厂商提供的 wheel，应改为安装该 wheel，并确认版本为 1.9.3。不要把厂商 wheel、机器人序列号、IP、标定文件或凭据提交到 GitHub。

开发/测试依赖可选安装：

```bash
./scripts/bootstrap_ubuntu.sh --dev
```

OpenPI 官方客户端可选安装：

```bash
python -m pip install -e '.[m4]'
```

Force Policy 不是只读部署依赖，也不随本仓库发布。需要研究其数据格式或部署方法时，应单独克隆上游仓库并固定 commit：

```bash
git clone https://github.com/force-policy/ForcePolicy.git
```

## 克隆后的无硬件验证

```bash
source .venv/bin/activate

# 配置与 fake smoke；不会访问硬件
./scripts/smoke.sh

# 完整测试；只使用 fake/injected SDK 和 127.0.0.1
./scripts/run_tests.sh -q
```

本次发布前基线为 `426 passed`。

常用 fake 命令：

```bash
vla-doctor --json
vla-shadow --duration 2 --json
vla-record --duration 2 --output /tmp/vla_raw --min-free-gb 0 --json
vla-m3 --duration 2 --output /tmp/m3.jsonl --json
```

## 第一次连接 Flexiv：只读 10 秒

执行前必须满足：现场有人、急停可触达、工作区清空、序列号与机器人网卡确认无误、没有其他 RDK/ROS2 writer 进程。

```bash
export VLA_INFRA_ALLOW_REAL_HARDWARE=1

vla-m5-readonly \
  --robot-serial '<真实序列号>' \
  --network-interface '<机器人网卡名，例如 enp3s0>' \
  --duration 10 \
  --rate-hz 10 \
  --ack I_ACKNOWLEDGE_FLEXIV_READONLY_NO_MOTION \
  --json | tee m5_field01_readonly.json
```

该命令只使用以下 RDK API：

- `flexivrdk.Robot(...)`
- `robot.states()`
- `robot.fault()`
- `robot.operational()`

它不会清除 fault。若失败，保留完整输出，不要转而运行 Flexiv 官方示例——官方 `basics1` 示例包含 `ClearFault()` 和 `Enable()`，不是只读测试。

测试完成后关闭当前终端或执行：

```bash
unset VLA_INFRA_ALLOW_REAL_HARDWARE
```

把 `m5_field01_readonly.json`、工控机系统版本、RDK 版本、网卡信息和测试时间交给 Leader 审查。首次成功后也不要自行延长到 30 分钟或尝试运动。

## 配置管理

仓库中的 `configs/*.template.yaml` 和 `configs/*.example.yaml` 只有模板。真实配置应复制到 `configs/local/`：

```bash
mkdir -p configs/local
cp configs/station.real.template.yaml configs/local/station.real.yaml
cp configs/safety.real.template.yaml configs/local/safety.real.yaml
```

`configs/local/` 已被 Git 忽略。以下内容不得进入 Git：

- 机器人/相机序列号和控制网络地址；
- 标定文件、TCP/payload 参数及其现场签名；
- API token、SSH key、VPN 凭据；
- 录制数据、日志和现场证据。

## 代码结构

```text
configs/                 示例和实机配置模板
docs/                    M1–M5 设计与验收矩阵
exchange/leader/         按时间编码的审查、实现和现场门禁记录
reference/               外部资料说明；第三方仓库不打包
scripts/                 安装、smoke、测试和发布前检查
src/vla_infra/
  cli/                   doctor/shadow/record/M3/M5只读入口
  contracts/             observation/action/pose 数据契约
  devices/               fake、RealSense、Flexiv RDK、GN01 adapter
  execution/             authority、安全门、插值、executor、supervisor
  policy/                WebSocket/OpenPI client/server/worker
  recording/             原始 episode、校验、回放和 LeRobot draft
  runtime/               collection、M3、M5 生命周期
  sync/                  环形缓冲与多传感器同步
tests/                   单元和集成测试（不访问真实硬件）
```

## 推送 GitHub 前

建议先使用 **private repository**：仓库虽然不应包含密钥和真实配置，但包含系统架构、故障处理和现场部署流程。确认组织的开源与安全策略后，再决定是否公开。

```bash
./scripts/release_check.sh
git status --short
git diff --check
```

确认输出中没有真实配置、数据或大文件，再执行自己的 `git add/commit/push`。仓库没有自动运行或接触硬件的 GitHub Actions；CI 仅运行无硬件测试。

## 距离实机运动仍缺少

1. 10 秒、1 分钟、30 分钟只读证据；
2. GN01 真实只读宽度 provider；
3. 双相机序列号、标定 hash、同步和断流验证；
4. 工控机实际数据盘 30 分钟压力测试；
5. 从真实 RDK Tool/ToolParams 独立读取并核验 TCP/payload；
6. Stop、断网、进程崩溃、E-stop 故障矩阵；
7. 经审查的现场运动 CLI 和新的 Leader Go 决策。

在这些条件完成前，`allow_real_motion` 必须保持 `false`。
