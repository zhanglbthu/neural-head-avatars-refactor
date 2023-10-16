from vht.model.tracking import FlameTracker as Tracker
from vht.data.video import VideoDataset
from vht.util.log import get_logger
from vht.util.video_to_dataset import Video2DatasetConverter

from argparse import ArgumentParser
from configargparse import ArgumentParser as ConfigArgumentParser
import configparser
from pathlib import Path

logger = get_logger("vht", root=True)


def read_camera(config_file):
    # 创建ConfigParser对象并读取配置文件
    config = configparser.ConfigParser() 
    config.read(config_file)
    # 使用config对象获取camera模块下的camera_names参数值
    try:
        camera_names = config.get("camera", "camera_names").split(",")
        root_path = config.get("camera", "root_path")
        return camera_names, root_path
    except (configparser.NoSectionError, configparser.NoOptionError):
        # 处理找不到section或option的情况
        print("No camera_names found in config file.")
        return []

def main():
    parser = ArgumentParser()
    parser = Tracker.add_argparse_args(parser)

    parser = ConfigArgumentParser(parents=[parser], add_help=False)
    parser.add_argument("--config", required=True, is_config_file=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--video", required=False)
    parser.add_argument(
        "--downscale_factor",
        type=float,
        default=1.0,
        help="Factor by which the input video is scaled to a lower resolution."
        " Mind that final pixel resolution is rounded to integer numbers.",
    )

    args = parser.parse_args()
    args_dict = vars(args)
    
    # data_path = Path(args.data_path)
    # if not data_path.exists() or args.video is not None:
    #     converter = Video2DatasetConverter(args.video, args.data_path)
    #     converter.extract_frames()
    #     converter.annotate_landmarks()

    logger.info(
        f"Start tracking with the following"
        f" configuration: \n {parser.format_values()}"
    )

    if args.downscale_factor < 1.0:
        raise ValueError("Downscale factor of inputs is supposed to be >= 1.")
    
    datasets = []
    camera_names, root_path = read_camera(args.config)

    for camera_idx in camera_names:
        # 取camera_index的最后两位
        data_path = Path(root_path) / camera_idx[-2:]
        datasets.append(VideoDataset(data_path, args.downscale_factor, camera_idx))
    
    tracker = Tracker(datasets, **args_dict)
    tracker.optimize()


if __name__ == "__main__":
    main()
