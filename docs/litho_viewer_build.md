# litho_viewer 使用说明

## 打包（一次性）

```bash
cd /Users/wyh/Desktop/学校/Litho_cpp
python3 -m PyInstaller --onefile --name litho_viewer litho_viewer.py
# 产物: dist/litho_viewer (macOS) 或 dist/litho_viewer.exe (Windows)
```

## 命令行用法

### matrix 模式 — 显示灰度/热图

```bash
litho_viewer --mode matrix \
    --files mask.txt aerial.txt wafer.txt \
    --labels Mask Aerial Wafer \
    --cmaps gray inferno gray \
    --title "光刻结果" \
    --save result.png
```

### profile 模式 — 1D 剖面曲线

```bash
litho_viewer --mode profile \
    --files profile_mask.txt profile_aerial.txt profile_wafer.txt \
    --labels Mask Aerial Wafer \
    --threshold 0.3 \
    --save profile.png
```

### ep 模式 — EP 点可视化

```bash
litho_viewer --mode ep \
    --files eps_others.txt \
    --mask mask.txt \
    --title "EP Select" \
    --save ep.png
```

### compare 模式 — 并排对比 + 差值图

```bash
litho_viewer --mode compare \
    --files before.txt after.txt \
    --labels Before After \
    --save diff.png
```

## 参数速查

| 参数 | 说明 |
|------|------|
| `--mode matrix/profile/ep/compare` | 可视化模式 |
| `--files a.txt b.txt ...` | 输入 txt 文件列表 |
| `--labels A B ...` | 对应标签 |
| `--cmaps gray inferno ...` | colormap（每个文件一个） |
| `--mask mask.txt` | ep 模式下的背景掩模 |
| `--title "标题"` | 图形总标题 |
| `--save out.png` | 保存 PNG（不加则弹窗） |
| `--dpi 200` | 保存分辨率 |
| `--threshold 0.3` | profile 模式阈值线 |

## 支持的 txt 格式

| 格式 | 说明 |
|------|------|
| 矩阵 txt | 每行空格分隔浮点数，首行可含 `# rows=N cols=M` |
| 剖面 txt | 单列 y 值，或两列 (x, y) |
| EP txt | 4 列：`y x w_epe w_meef` |

## C++ 中调用

```cpp
// 保存 PNG（不弹窗）
std::system("../dist/litho_viewer --mode matrix \
    --files output.txt --labels Result --save out.png");

// 或弹交互窗口
std::system("../dist/litho_viewer --mode matrix \
    --files output.txt --labels Result");
```
