# TianMouCV Batch Processing

自动化批量处理 TianMouCV 多数据集的脚本工具。

## 文件说明

| 文件 | 作用 |
|---|---|
| `batch_datasets.yaml` | 配置文件，列出所有要处理的数据集 |
| `batch_process.py` | 单数据集完整处理流程（Step 3-6 的自动化封装） |
| `run_batch.py` | 并行调度器，自动利用多核 CPU 并行处理多个数据集 |

## 依赖

```bash
conda activate tianmoucv
pip install pyyaml
```

## 使用方法

### 1. 修改 `batch_datasets.yaml`

根据实际数据集名称修改 `datasets` 列表：

```yaml
datasets:
  - 0530_extreme1
  - 0530_extreme2
  # ... 添加更多
```

### 2. 确认路径

默认配置假设：
- 工作目录：`/home/nvidia/Desktop/cxr_multi_sensor/yyf_new`
- 原始数据：`/projects/<dataset_name>`
- 输出目录：`/projects/calib_data/output_<dataset_name>`

如有不同，修改 `batch_datasets.yaml` 中的对应字段，或在命令行覆盖。

### 3. 预览（不执行）

```bash
python run_batch.py --dry-run
```

### 4. 执行批量处理

```bash
python run_batch.py
```

脚本会自动：
- 创建输出目录（`sudo mkdir -p` + `sudo chmod 777`）
- 依次执行 `post_process.py` → `export_aligned_frames.py` → `calib/depth_0408.py`
- 每个数据集的日志保存到 `logs/<dataset_name>.log`
- 全部完成后打印汇总结果

# 预览（不会实际执行）
python run_batch.py --dry-run

### 5. 指定并行 worker 数量

默认使用全部 CPU 核心。可手动指定：

```bash
python run_batch.py --workers 4
```

## 处理流程（每数据集）

```
Step 1: sudo mkdir -p /projects/calib_data/output_<name>
Step 2: python3 post_process.py --root /projects/<name> -o output_<name>.json
Step 3: python3 export_aligned_frames.py --root /projects/<name> --index output_<name>.json --output .../output_<name> --use-raw-depth
Step 4: python3 calib/depth_0408.py --base .../output_<name> --index output_<name>.json
```

## 命名规范

所有文件/目录名称由 `config.yaml` 中的数据集名称自动派生，保证命名一致性：

| config 中的名称 | 原始数据路径 | JSON 文件 | 输出目录 |
|---|---|---|---|
| `0530_extreme1` | `/projects/0530_extreme1` | `output_0530_extreme1.json` | `/projects/calib_data/output_0530_extreme1` |
| `0530_extreme2` | `/projects/0530_extreme2` | `output_0530_extreme2.json` | `/projects/calib_data/output_0530_extreme2` |

## 注意事项

- **命名一致性**：每一步的 `--index` 和 `--output` 使用统一的派生名称，不再出现前后不一致的问题
- **并行安全**：每个数据集独立处理，无共享状态，可安全并行
- **故障恢复**：单个数据集失败不影响其他数据集；失败的任务日志在 `logs/` 中可查
