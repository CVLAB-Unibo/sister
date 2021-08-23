from pathlib import Path
import rich
import cv2
import numpy as np
import click
import shutil


class SisterDataset:
    LEVEL_TO_HEIGHT_MAP = {"0": "1cm", "1": "5cm", "2": "10cm"}

    def __init__(
        self,
        scenes_folder,
        gt_folder,
    ) -> None:

        self.scenes_folder = Path(scenes_folder)
        self.gt_folder = Path(gt_folder)
        self.scenes_map = {}
        scenes_subfolders = self.subfolders(self.scenes_folder)
        gts_map = self.build_gts_map(self.gt_folder)

        self.object_names = set()

        for f in scenes_subfolders:
            self.scenes_map[f.name] = {"images": self.build_levels_map(f), "gt": None}
            if f.name in gts_map:
                self.scenes_map[f.name]["gt"] = gts_map[f.name]
                self.object_names.add(f.name)

    def images_folder(self, object_name, level="1", baseline="100"):
        return self.scenes_map[object_name]["images"][level][baseline]

    def object_baselines(self, object_name, level):
        return list(self.scenes_map[object_name]["images"][level].keys())

    def load_image(self, object_name, level="1", baseline="100", direction="center"):
        images_folder = self.images_folder(object_name, level, baseline)
        image = cv2.imread(str(Path(images_folder) / f"00000_{direction}.png"))
        return image

    def load_prediction(
        self,
        object_name,
        level="1",
        baseline="100",
        prediction_name="FULL_mccnn_refine",
    ):
        images_folder = self.images_folder(object_name, level, baseline)
        output_folder = images_folder / "output"
        prediction_file = str(output_folder / f"{prediction_name}.png")
        depth = cv2.imread(prediction_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        return depth

    def gt_levels(self, object_name):
        return list(self.scenes_map[object_name]["gt"].keys())

    def gt_file(self, object_name, level="1"):
        return self.scenes_map[object_name]["gt"][level]

    def load_gt(self, object_name, level="1"):
        gt_file = self.gt_file(object_name, level)
        depth = cv2.imread(str(gt_file), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        return depth

    def subfolders(self, folder):
        return [x for x in Path(folder).glob("*") if x.is_dir()]

    def innerfiles(self, folder):
        return [x for x in Path(folder).glob("*") if not x.is_dir()]

    def build_levels_map(self, folder):
        levels_subfolders = self.subfolders(folder)
        levels_map = {}
        for level_subfolder in levels_subfolders:
            name = level_subfolder.stem
            _, level_id, baseline = name.split("_")
            if level_id not in levels_map:
                levels_map[level_id] = {}

            levels_map[level_id][baseline] = level_subfolder
        return levels_map

    def build_gts_map(self, folder):
        files = self.innerfiles(folder)
        gts_map = {}
        for f in files:
            name = f.stem
            chunks = name.split("_")
            object_name = "_".join(chunks[:-1])
            level_id = chunks[-1]
            if object_name not in gts_map:
                gts_map[object_name] = {}
            gts_map[object_name][level_id] = f
        return gts_map


def mix_images(i0, i1, alpha=0.5):
    image = (i0 * alpha + i1 * (1 - alpha)) / 2.0
    image = image.astype(np.uint8)
    return image


def colorize(depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = (depth * 255).astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
    return depth


def create_if_none(path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


@click.command("debug")
@click.option("--scenes_folder", default="objects_full_scenes")
@click.option(
    "--gt_folder", default="objects_full_scenes_GT/objects_full_scenes_gtmanual"
)
@click.option("--object_name", default="arduino")
@click.option("--level", default="1")
@click.option("--baseline", default="010")
@click.option("--prediction", default="FULL_mccnn_refine")
def debug(scenes_folder, gt_folder, object_name, level, baseline, prediction):

    dataset = SisterDataset(scenes_folder=scenes_folder, gt_folder=gt_folder)

    image = dataset.load_image(object_name, level, baseline)
    depth = dataset.load_gt(object_name, level=level)
    pred = dataset.load_prediction(
        object_name, level, baseline, prediction_name=prediction
    )

    depth = colorize(depth)
    pred = colorize(pred)
    mix = mix_images(image, depth)

    cv2.imshow("image", image)
    cv2.imshow("gt", depth)
    cv2.imshow("pred", pred)
    cv2.imshow("mix", mix)
    cv2.waitKey(0)


@click.command("export")
@click.option("--scenes_folder", default="objects_full_scenes")
@click.option(
    "--gt_folder", default="objects_full_scenes_GT/objects_full_scenes_gtmanual"
)
@click.option("--output_folder", required=True)
def export(scenes_folder, gt_folder, output_folder):

    dataset = SisterDataset(scenes_folder=scenes_folder, gt_folder=gt_folder)

    output_folder = create_if_none(Path(output_folder))

    for object_name in dataset.object_names:
        object_folder = create_if_none(output_folder / object_name)

        levels = sorted(dataset.gt_levels(object_name))
        for level in levels:
            height = SisterDataset.LEVEL_TO_HEIGHT_MAP[level]
            level_folder = create_if_none(object_folder / height)
            baselines = dataset.object_baselines(object_name, level)
            for baseline in baselines:
                baseline_distance = f"{baseline}mm"
                baseline_folder = create_if_none(level_folder / baseline_distance)

                images_folder = dataset.images_folder(object_name, level, baseline)
                images_files = list(sorted(images_folder.glob("*.png")))

                for source_filename in images_files:
                    dest_filename = baseline_folder / source_filename.name.replace(
                        "00000_", ""
                    )
                    shutil.copy(source_filename, dest_filename)

            source_gt_file = dataset.gt_file(object_name, level)
            dest_gt_file = level_folder / "gt_depth.exr"
            shutil.copy(source_gt_file, dest_gt_file)


@click.group()
def cli():
    pass


cli.add_command(debug)
cli.add_command(export)

if __name__ == "__main__":
    cli()
