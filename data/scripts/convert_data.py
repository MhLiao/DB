import shutil

import xmltodict

from os import PathLike
from pathlib import Path
from typing import List, Dict, Tuple, Union


class Annotation:
    def __init__(self, ytl: int, ybr: int, xtl: int, xbr: int, lang: str) -> None:
        self.ytl, self.ybr, self.xtl, self.xbr, self.lang = ytl, ybr, xtl, xbr, lang


class MalformedTaskException(Exception):
    pass


PathType = Union[Path, PathLike]
Sample = Tuple[PathType, List[Annotation]]  # image_path, annotations
RawBox = Dict[str, str]


def convert(src: PathType, dst: PathType) -> None:
    (dst / 'images').mkdir(exist_ok=True)
    (dst / 'gts').mkdir(exist_ok=True)
    for task in src.iterdir():
        if not task.is_dir():
            print('Unexpected: ' + str(task))
            continue
        try:
            samples = TaskParser(task).parse()
            write_task(dst, task.name, samples)
        except MalformedTaskException as e:
            print(type(e), e)


class TaskParser:

    def __init__(self, task_path: PathType) -> None:
        self.task_path = task_path
        self.lang = None  # initialized on parsing

    def parse(self) -> List[Sample]:
        print(f'Parsing task: {self.task_path}')
        annotations, images = None, None
        for thing_path in self.task_path.iterdir():
            if thing_path.is_file() and thing_path.suffix == '.xml':
                annotations = self.parse_xml(thing_path)
            elif thing_path.is_dir():
                images = [img.absolute() for img in thing_path.iterdir() if
                          img.suffix in ['.jpg', '.jpeg', '.png', '.bmp']]
            else:
                print('Unexpected: ' + str(thing_path))
        if annotations and images and len(images) == len(annotations):
            print(f'There are {len(images)} images in task.')
            sorted_annotations = [annotations[k] for k in sorted(annotations.keys())]
            return list(zip(sorted(images), sorted_annotations))
        else:
            raise MalformedTaskException(f'Task is malformed: {self.task_path}')

    def parse_xml(self, xml_file: PathType) -> Dict[str, List[Annotation]]:
        raw_dict = xmltodict.parse(xml_file.open(encoding='utf-8').read())
        self.lang = 'AR' if 'AR' in raw_dict['annotations']['meta']['task']['name'].upper() else 'PR'
        raw_task_annotations = raw_dict['annotations']['image']
        if type(raw_task_annotations) == list:  # multiple images in task
            return {img_ann['@name']: self.to_image_annotations(img_ann['box']) for img_ann in raw_task_annotations}
        return {raw_task_annotations['@name']: self.to_image_annotations(raw_task_annotations['box'])}

    def to_image_annotations(self, raw_image_annotations: Union[List[RawBox], RawBox]) -> List[Annotation]:
        if type(raw_image_annotations) == list:  # multiple boxes in image
            return [self.to_annotation_obj(clip_ann) for clip_ann in raw_image_annotations]
        return [self.to_annotation_obj(raw_image_annotations)]

    def to_annotation_obj(self, clip_annotation: RawBox) -> Annotation:
        return Annotation(*map(lambda coord: round(float(clip_annotation['@' + coord])), ['ytl', 'ybr', 'xtl', 'xbr']),
                          lang=self.lang)


def write_task(dst_path: PathType, task_name: str, samples: List[Sample]) -> None:
    print(f'Converting {task_name}.')
    for img_path, img_annotations in samples:
        img_name = task_name + '_' + img_path.name
        img_dst_path = dst_path / 'images' / img_name
        shutil.copy(img_path, img_dst_path)
        print(f'    Copied "{img_path.name}" as "{img_dst_path.name}" into destination.')
        gt_dst_path = dst_path / 'gts' / (img_name + '.txt')
        with gt_dst_path.open(mode='w') as f:
            for a in img_annotations:
                f.write(f'{a.xtl},{a.ybr},{a.xtl},{a.ytl},{a.xbr},{a.ytl},{a.xbr},{a.ybr},{a.lang},AAA\n')
        print(f'    Created annotations file "{gt_dst_path.name}" in destination.')


def main() -> None:
    src = Path(r'/mnt/data/idan/DetectionData')
    dst = Path(r'/mnt/data/idan')
    convert(src, dst)


if __name__ == '__main__':
    main()
