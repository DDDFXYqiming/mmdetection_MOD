"""统一命令行入口：mmdet-scripts。"""

import argparse
import sys

from .coco.cleanup import delete_json_in_coco_subdirs, delete_orphan_json
from .coco.convert import convert_to_coco
from .coco.split import batch_split, split_dataset
from .filter.select import select_by_labels_and_distance
from .tools.media import images_to_video
from .tools.pkl_inspect import inspect_pkl
from .visualize.annotations import visualize_annotations
from .visualize.predictions import visualize_pkl_compare, visualize_pkl_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmdet-scripts",
        description="mmdetection_MOD 数据管线：COCO 转换、数据分割、筛选、可视化与导出。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # split：按时间戳分组分割
    p_split = sub.add_parser("split", help="按时间戳分组分割为 train/val/test")
    p_split.add_argument("json_dir", help="原始 jpg+json 目录")
    p_split.add_argument("output_dir", help="输出目录（生成 coco/train2017 等）")
    p_split.add_argument("--train-ratio", type=float, default=0.7)
    p_split.add_argument("--val-ratio", type=float, default=0.2)
    p_split.add_argument("--max-time-diff", type=float, default=60, help="分组最大时间差（分钟）")
    p_split.add_argument("--no-sort-groups", action="store_true", help="不按组大小排序（保持时间顺序）")

    # batch：分片
    p_batch = sub.add_parser("batch", help="按固定数量分批复制")
    p_batch.add_argument("json_dir")
    p_batch.add_argument("output_dir")
    p_batch.add_argument("--batch-size", type=int, default=500)

    # convert：COCO 转换
    p_conv = sub.add_parser("convert", help="原始标注 json 转换为 COCO 格式")
    p_conv.add_argument("json_dir")
    p_conv.add_argument("output_dir")
    p_conv.add_argument("--output", default="instances.json", help="输出文件名")
    p_conv.add_argument(
        "--variant",
        choices=["standard", "mini", "keypoints", "point-box"],
        default="standard",
        help="转换变体：standard=3类无关键点；mini=4类；keypoints=4类+关键点；point-box=关键点转小框",
    )

    # cleanup
    p_clean = sub.add_parser("cleanup", help="清理数据集")
    p_clean.add_argument("folder")
    p_clean.add_argument("--orphan-json", action="store_true", help="删除没有对应图片的 json")
    p_clean.add_argument("--coco-json", action="store_true", help="删除 coco 子目录（train2017 等）中的 json")

    # select
    p_sel = sub.add_parser("select", help="按目标类型与距离阈值筛选数据")
    p_sel.add_argument("origin_folder")
    p_sel.add_argument("output_folder")
    p_sel.add_argument("--types", nargs="*", default=None,
                       help="目标类型，默认 Cyclist/Ped/FullCar")
    p_sel.add_argument("--max-distance", type=float, default=50)
    p_sel.add_argument("--interval", type=int, default=4, help="每隔多少个 json 处理一次")
    p_sel.add_argument("--formula", choices=["corrected", "legacy"], default="corrected",
                       help="距离公式：corrected=修正后（默认）；legacy=原版笔误公式")

    # visualize
    p_viz = sub.add_parser("visualize", help="COCO 标注可视化")
    p_viz.add_argument("annotation_file")
    p_viz.add_argument("image_folder")
    p_viz.add_argument("--output", default="./annotated_images")
    p_viz.add_argument("--no-keypoints", action="store_true", help="不绘制关键点")

    p_vizpkl = sub.add_parser("viz-pkl", help="pkl 预测结果可视化")
    p_vizpkl.add_argument("pkl_path")
    p_vizpkl.add_argument("--output", default="./pkl_cv")
    p_vizpkl.add_argument("--score-threshold", type=float, default=0.5)
    p_vizpkl.add_argument("--alpha", type=float, default=0.65)
    p_vizpkl.add_argument("--compare", action="store_true", help="与标注拼接对比")
    p_vizpkl.add_argument("--annotation-file", default=None)
    p_vizpkl.add_argument("--image-folder", default=None)

    # to-video
    p_video = sub.add_parser("to-video", help="图片序列生成视频")
    p_video.add_argument("image_folder")
    p_video.add_argument("--output", default="demo_car.mp4")
    p_video.add_argument("--fps", type=int, default=10)

    # read-pkl
    p_pkl = sub.add_parser("read-pkl", help="查看 pkl 预测结果")
    p_pkl.add_argument("pkl_path")
    p_pkl.add_argument("--limit", type=int, default=3)

    # export-onnx
    p_onnx = sub.add_parser("export-onnx", help="pth → onnx 导出（需要 torch + mmdetection 环境）")
    p_onnx.add_argument("checkpoint")
    p_onnx.add_argument("--output", default="./yolox.onnx")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "split":
            split_dataset(
                args.json_dir,
                args.output_dir,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                max_time_diff_minutes=args.max_time_diff,
                sort_groups_by_size=not args.no_sort_groups,
            )
        elif args.command == "batch":
            batch_split(args.json_dir, args.output_dir, batch_size=args.batch_size)
        elif args.command == "convert":
            convert_to_coco(args.json_dir, args.output_dir, args.output, variant=args.variant)
        elif args.command == "cleanup":
            if args.orphan_json:
                delete_orphan_json(args.folder)
            if args.coco_json:
                delete_json_in_coco_subdirs(args.folder)
            if not args.orphan_json and not args.coco_json:
                print("请指定 --orphan-json 或 --coco-json")
                return 2
        elif args.command == "select":
            select_by_labels_and_distance(
                args.origin_folder,
                args.output_folder,
                target_types=args.types,
                max_distance=args.max_distance,
                interval=args.interval,
                formula=args.formula,
            )
        elif args.command == "visualize":
            visualize_annotations(
                args.annotation_file,
                args.image_folder,
                output_folder=args.output,
                draw_keypoints=not args.no_keypoints,
            )
        elif args.command == "viz-pkl":
            if args.compare:
                if not args.annotation_file or not args.image_folder:
                    print("--compare 需要 --annotation-file 与 --image-folder")
                    return 2
                visualize_pkl_compare(
                    args.pkl_path,
                    args.annotation_file,
                    args.image_folder,
                    output_dir=args.output,
                    score_threshold=args.score_threshold,
                    alpha=args.alpha,
                )
            else:
                visualize_pkl_predictions(
                    args.pkl_path,
                    output_dir=args.output,
                    score_threshold=args.score_threshold,
                    alpha=args.alpha,
                )
        elif args.command == "to-video":
            images_to_video(args.image_folder, video_name=args.output, fps=args.fps)
        elif args.command == "read-pkl":
            inspect_pkl(args.pkl_path, limit=args.limit)
        elif args.command == "export-onnx":
            from .export.onnx import export_onnx

            export_onnx(args.checkpoint, args.output)
    except KeyboardInterrupt:
        print("\n已中断")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
