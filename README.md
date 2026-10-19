# SlideSparse-Diffusion

Reusable structured-sparse W8A8 diffusion inference and research code for NVIDIA SM80.

把 **PixArt-Sigma、PixArt-α、FLUX、DiT-XL、SD3-Medium、SD3.5-Large** 的适配放在同一个库中，共用校准、权重银行、逐步策略、仿射补偿、计时和指标代码。模型结构差异集中在 `models/`，不再保留多套编号工作版本。

这是已有研究实现的源码整理与公开发布。原始权重、已校准银行、逐图数据和内部任务日志分别保留；安装、查看模型列表不会自动运行实验。

## 模型与默认设置

| 统一名称 | 分辨率 | 调度与步数 | 适配 |
| --- | --- | --- | --- |
| `sigma` | 1024×1024 | DPM-Solver++，20步 | PixArt fused QKV/KV、FFN |
| `pixart-alpha` | 256×256 | DPM-Solver++，20步 | 复用 PixArt 路由，原生256模型 |
| `flux` | 512×512 | FlowMatch Euler，50步 | 双流/单流块，保护 attention 贡献 |
| `dit-xl` | 512×512；另支持256模型 | DDIM，50步 | 原生 class/null CFG、learned-sigma 输出 |
| `sd3-medium` | 512×512 | FlowMatch Euler，25步 | MMDiT、三路文本条件 |
| `sd3.5-large` | 512×512 | FlowMatch Euler，25步 | 共用 MMDiT，保留原生 Q/K RMSNorm |

DiT 的适配状态与完整四档实验结论分开。已有实验使用各自冻结源码与条件；本次源码整理没有重新运行模型实验，也不把历史结果标成公开副本的新测量。

## 代码组织

```text
sdacc/
  diffusion_research/
    models/             # 模型结构与投影路由
    configs/            # 六个模型的可配置本地路径模板
    runtime.py          # 共用运行时与逐步选择
    prepare.py          # 曲率、权重银行、固定策略
    calibration.py      # 每档自身的未恢复轨迹校准
    affine.py           # FP32 离线 fold 与运行时参数选择
    runner.py           # 推理和正式配对计时
    image_metrics.py    # 解码与图像指标
    integrations/       # 缓存/调度/稀疏注意力研究接入
  w8a8_slidesparse/      # 既有0.3.3算子源码、静态profile、CUTLASS依赖标识
scripts/build_native.py # 显式编译本机ABI的SM80扩展
docs/                   # 使用协议、数据布局、研究边界
provenance/             # 导出来源和原始SHA；无内部任务流水账
```

算子支持 `2:4`、`4:6`、`6:8`，动态 per-token INT8 激活、per-output-channel 权重尺度、BF16 输出。滑动展开与量化在同一前端中完成，打包在离线阶段执行。完整投影、完整去噪和图像质量是不同层次的结果。

重建采用 OBS-Diff 风格的固定顺序共享曲率与联合 INT8 补偿；这里的实现**不标作官方 ExactOBC 复现**。attention core/out 保持保护；FLUX 的混合输出保留未稀疏的 attention 分量。

## 安装与查看

仅查看目录，无需 Torch 或 GPU：

```bash
git clone https://github.com/WangSibothunder/SlideSparse-Diffusion.git
cd SlideSparse-Diffusion
python -m pip install -e .
python -m sdacc.diffusion_research models
python -m sdacc.diffusion_research.integrations
```

运行稀疏模型需要 Linux、NVIDIA SM80 GPU、匹配的 Torch/CUDA ABI、CUDA 编译器及 CUTLASS 源码。已使用的算子环境为 Torch 2.5.1+cu121 / Triton 3.1.0 / nvcc 12.4 / C++ ABI0；不同环境需重新编译。公开打包副本未在新的GPU环境重测。

```bash
# 先按本机CUDA环境安装Torch，然后安装运行依赖。
python -m pip install -e '.[inference,metrics]'
git clone https://github.com/NVIDIA/cutlass.git external/cutlass
git -C external/cutlass checkout 0b55a2f691d69981583568fd9eb69687b1f0de8a
python scripts/build_native.py --cutlass external/cutlass
```

编译是显式操作，包安装不编译CUDA。模型需自行按其许可下载到本机；仓库不包含模型权重，也不替代模型访问权限。

## 复用已有校准与推理

配置模板位于 `sdacc/diffusion_research/configs/`。设置自己的 checkpoint、conditions、conditioning、bank、policy 与 coefficients 路径；支持相对配置文件的路径、`~` 及环境变量。保留既有结果目录，执行前选择新的输出目录。

```python
from sdacc.diffusion_research import definition, inventory, build

definition("sd3.5-large")  # 只返回模型元数据
# inventory(name, native_transformer, resolution, text_length)
# build(name, config, output) 显式加载本模型已有银行与系数
```

下列命令会实际推理，仅在已准备好对应模型资产时执行：

```bash
python -m sdacc.diffusion_research run \
  --model sd3.5-large \
  --config sdacc/diffusion_research/configs/sd3.5-large.json \
  --stage infer --policy priority_s25 --affine \
  --gpu 0 --output outputs/sd35-infer
```

`run` 另提供 `calibrate`、`benchmark`、`profile`、`decode`、`score`、`study` 显式阶段。已有银行和系数可直接复用；推理与必要复测速不要求重跑校准和质量。

Sigma/FLUX 的历史银行还有各自既定 manifest/slot-plan 格式，不能仅改文件名后互换。旧插件质量回填和 Sigma CEM 桥接依赖原外部研究包，见[研究接入说明](sdacc/diffusion_research/integrations/README.md)。图像指标使用固定资产布局，见[使用与数据约定](docs/USAGE.md)。

## 计时与研究范围

主加速比分母是同次配对的 **default compiled BF16 Dense**。同样采用完整图优化的 Dense 对照另列，通用图优化不计作 Sparse 的独立贡献。正式去噪计时覆盖策略选择、必要 H2D/D2D 和参数切换；仿射系数离线拟合、运行时不拟合。

完整研究协议采用目标15/25/35/45%、每档100cal、4pilot、64eval；实际结构稀疏率单列。PSNR 与 Gaussian11/σ1.5 wSSIM 对照同条件 Dense，CLIP 为 ViT-B/16，ImageReward 使用 v1.0。详情见[实验口径](docs/PROTOCOL.md)。

缓存、DuCa、CEM、SpargeAttn 与 CacheQuant 为可选研究接入；负结果和 hold 状态保留，不作为默认开启项或普遍加速承诺。

## 来源与许可

本仓库以 GPL-3.0 发布（包含 DuCa 衍生适配）；第三方文件保留原许可证。SlideSparse/SpargeAttn 的 Apache-2.0、NVIDIA CUTLASS/NVML 的 BSD 许可及来源见 [NOTICE](NOTICE)、[LICENSES](LICENSES/) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。模型及外部插件资产遵循各自许可。
