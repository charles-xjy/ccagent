# CubeSandbox 部署指南

> 目标：在 10.145.x.x 服务器上部署 CubeSandbox，为 ccagent 的代码执行提供 KVM 硬件级隔离沙箱。

## 架构说明

```
ccagent（你的 Python 服务）
    │
    │  调用 e2b-code-interpreter SDK（本地 pip 包）
    │  发 HTTP 请求到 CubeSandbox API
    ▼
CubeSandbox（运行在 10.145 服务器上，端口 3000）
    └── 每次执行测试，启动一个独立 KVM MicroVM
    └── 测试跑完/崩了，VM 销毁，服务器不受影响
```

---

## 前置要求

| 项目 | 要求 |
|------|------|
| 架构 | x86_64 |
| 系统 | Ubuntu 22.04 / 24.04（推荐），或 OpenCloudOS 9 / TencentOS 4 |
| 磁盘 | `/data/cubelet` 目录至少 50GB，推荐 200GB+ |
| 文件系统 | 支持 XFS |
| glibc | ≥ 2.35 |
| 权限 | root |

---

## 第一步：安装 PVM 内核

CubeSandbox 需要专用的 PVM（Para-Virtual Machine）内核来驱动 KVM 沙箱。

### Ubuntu 系统（DEB）

**第一步：下载 PVM 内核包**

打开浏览器访问：https://cnb.cool/CubeSandbox/CubeSandbox/-/releases

找到最新 release（如 v0.3.0），在附件列表中找到文件名格式为：
```
linux-image-*cube.pvm.host*_amd64.deb
```
右键复制下载链接，然后在服务器上执行：

```bash
# 下载（替换为你复制的实际链接）
wget "https://cnb.cool/CubeSandbox/CubeSandbox/-/releases/download/v0.3.0/linux-image-*cube.pvm.host*_amd64.deb"

# 安装内核
dpkg -i linux-image-*cube.pvm.host*.deb

# 配置引导（自动设置为默认启动项）
KVER="$(ls /boot/vmlinuz-*cube.pvm.host* | sed 's|/boot/vmlinuz-||' | tail -1)"
curl -sL https://cnb.cool/CubeSandbox/CubeSandbox/-/git/raw/master/deploy/pvm/grub/host_grub_config.sh | bash

# 重启切换到 PVM 内核
reboot
```

### CentOS / TencentOS 系统（RPM）

```bash
# 1. 下载 RPM 内核包
wget "<kernel-*cube.pvm.host*.rpm 的下载链接>"

# 2. 安装
rpm -ivh --oldpackage kernel-*.rpm

# 3. 设置为默认启动项
grubby --info=ALL | grep -E "^kernel|^index"
grubby --set-default-index=<上面查到的新内核 index>
curl -sL https://cnb.cool/CubeSandbox/CubeSandbox/-/git/raw/master/deploy/pvm/grub/host_grub_config.sh | bash

# 4. 重启
reboot
```

### 验证内核安装成功

```bash
# 内核版本应包含 "cube.pvm.host"
uname -r

# 加载并验证 KVM 模块
modprobe kvm_pvm && lsmod | grep kvm

# 设置开机自动加载
echo 'kvm_pvm' > /etc/modules-load.d/kvm-pvm.conf
```

---

## 第二步：一键安装 CubeSandbox

```bash
curl -sL https://cnb.cool/CubeSandbox/CubeSandbox/-/git/raw/master/deploy/one-click/online-install.sh \
  | CUBE_PVM_ENABLE=1 MIRROR=cn bash
```

这个脚本会自动安装：
- CubeAPI（REST 网关，端口 3000）
- CubeMaster（调度器）
- Cubelet（节点调度）
- CubeHypervisor（KVM 管理）
- MySQL + Redis（通过 Docker Compose）

安装完成后验证服务是否正常：

```bash
curl http://localhost:3000/health
```

---

## 第三步：创建代码执行模板

模板是沙箱的基础镜像，agent 每次执行代码都从这个模板启动一个新 VM。

```bash
# 创建模板（使用官方 Python 代码执行镜像）
cubemastercli tpl create-from-image \
  --image cube-sandbox-cn.tencentcloudcr.com/cube-sandbox/sandbox-code:latest \
  --writable-layer-size 1G \
  --expose-port 49999 \
  --expose-port 49983 \
  --probe 49999

# 查看创建进度（等待 status 变为 ready）
cubemastercli tpl watch --job-id <上面输出的 job_id>
```

创建成功后会输出 `template_id`，记录下来下一步用。

---

## 第四步：配置 ccagent 环境变量

在 ccagent 的启动脚本或 `.env` 文件中添加：

```bash
export E2B_API_URL="http://10.145.x.x:3000"   # 替换为实际服务器 IP
export E2B_API_KEY="e2b_000000"                # CubeSandbox 不校验此值，固定填这个
export CUBE_TEMPLATE_ID="<第三步拿到的模板ID>"
```

`nano_claude_code/core/sandbox.py` 会自动读取这些变量连接到 CubeSandbox，无需改代码。

---

## ccagent 侧依赖安装

```bash
pip install "e2b-code-interpreter>=2.4.1"
```

---

## 验证整体流程

```python
# 快速验证沙箱连通性
import os
os.environ["E2B_API_URL"] = "http://10.145.x.x:3000"
os.environ["E2B_API_KEY"] = "e2b_000000"
os.environ["CUBE_TEMPLATE_ID"] = "<模板ID>"

from e2b_code_interpreter import Sandbox

with Sandbox(template=os.environ["CUBE_TEMPLATE_ID"]) as sb:
    result = sb.run_code('print("hello from sandbox")')
    print(result.logs.stdout)  # 应输出 ['hello from sandbox\n']
```

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `nano_claude_code/core/sandbox.py` | 沙箱会话管理（单例，进程退出自动销毁） |
| `nano_claude_code/coder_agent/tools.py` | bash / run_python / write_file 等工具，全部走沙箱 |

## 参考

- [CubeSandbox GitHub](https://github.com/TencentCloud/CubeSandbox)
- [E2B Python SDK 文档](https://e2b.dev/docs)
