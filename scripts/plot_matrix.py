import argparse
from ast import parse
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

def _save_with_colorbar(image_path, save_path, cmap, dpi: int = 300, title: str = None):
        """用 imshow + colorbar 保存一张二维数组的可视化图，自带颜色条。

        - 关闭坐标轴；
        - 输出 dpi 由参数控制；
        - 保存后立即关闭 figure 防内存累积。
        """
        save_path = Path(save_path)
        # 如果传的是目录，自动补文件名
        if save_path.is_dir() or save_path.suffix == "":
            save_path = save_path / (Path(image_path).stem + ".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[plot] image_path = {image_path}")
        print(f"[plot] save_path  = {save_path}")

        image = np.loadtxt(image_path, comments="#")
        arr = np.asarray(image)
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(arr, cmap=cmap)
        ax.set_axis_off()
        if title:
            ax.set_title(str(title))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] saved: {save_path}")

if __name__ == "__main__":
  # python3 /Users/wyh/Desktop/学校/Litho_cpp/scripts/plot_matrix.py   --multi  --files /Users/wyh/Desktop/学校/Litho_cpp/result/wafer.txt /Users/wyh/Desktop/学校/Litho_cpp/result/result.txt /Users/wyh/Desktop/学校/Litho_cpp/result/binary_mask.txt  /Users/wyh/Desktop/学校/Litho_cpp/result/binarry_mask_OTSU.txt /Users/wyh/Desktop/学校/Litho_cpp/result/binary_new_mask.txt  --titles "wafer" "result" "binary" "binary_OTSU" "newbinary"  --cmaps  hot hot gray gray gray  --rows 2  --save_path ./result/multi_view.png

   parser = argparse.ArgumentParser(description="光刻仿真结果可视化")
   parser.add_argument("--image_path", type=str, help="输入文件路径")
   parser.add_argument("--save_path", type=str, help="输出文件路径")
   parser.add_argument("--cmap", type=str, default="hot", help="colormap")
   parser.add_argument("--dpi", type=int, default=300, help="输出 dpi")
   parser.add_argument("--title", type=str, default=None, help="标题")

   # ── 多图拼接子命令 ──
   parser.add_argument("--multi", action="store_true", help="多图拼接模式")
   parser.add_argument("--files", nargs="+", help="多图模式下，所有输入文件路径")
   parser.add_argument("--titles", nargs="+", default=None, help="每张图的标题")
   parser.add_argument("--cmaps", nargs="+", default=None, help="每张图的 colormap（默认全 hot）")
   parser.add_argument("--rows", type=int, default=1, help="多图模式下行数")
   args = parser.parse_args()

   if args.multi:
       # 多图模式
       assert args.files, "--multi 需要 --files <a.txt b.txt ...>"
       n = len(args.files)
       titles = args.titles or [None]*n
       cmaps  = args.cmaps  or [args.cmap]*n
       rows   = args.rows if args.rows > 0 else 1
       cols   = (n + rows - 1) // rows

       fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
       axes = np.array(axes).reshape(-1)   # 扁平化，方便索引
       for idx, (f, t, cm) in enumerate(zip(args.files, titles, cmaps)):
           arr = np.loadtxt(f, comments="#")
           im = axes[idx].imshow(arr, cmap=cm)
           axes[idx].set_axis_off()
           if t: axes[idx].set_title(t)
           fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
       # 关闭多余的子图位置
       for idx in range(n, len(axes)):
           axes[idx].set_axis_off()
       fig.tight_layout()
       save = args.save_path or "./multi_view.png"
       fig.savefig(save, dpi=args.dpi, bbox_inches="tight")
       plt.close(fig)
       print(f"[plot] multi saved: {save}")
   else:
       # 单图模式（原逻辑）
       _save_with_colorbar(args.image_path, args.save_path, args.cmap, args.dpi, args.title)
