import itertools
import os

from typing import List, Optional

import torch
import numpy as np
import tempfile
from collections import OrderedDict
from PIL import Image
from tabulate import tabulate
import json
import contextlib

import detectron2.utils.comm as comm
from detectron2.utils.file_io import PathManager
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOPanopticEvaluator

from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
import io
import math
from PIL import Image

from detectron2.solver.lr_scheduler import _get_warmup_factor_at_iter


import logging

logger = logging.getLogger(__name__)


class TF2WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Poly learning rate schedule used in TF DeepLab2.
    Reference: https://github.com/google-research/deeplab2/blob/main/trainer/trainer_utils.py#L23
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
        power: float = 0.9,
        constant_ending: float = 0.0,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.power = power
        self.constant_ending = constant_ending
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        if self.constant_ending > 0 and warmup_factor == 1.0:
            # Constant ending lr.
            if (
                math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
                < self.constant_ending
            ):
                return [base_lr * self.constant_ending for base_lr in self.base_lrs]
        if self.last_epoch < self.warmup_iters:
            return [
            base_lr * warmup_factor
            for base_lr in self.base_lrs
        ]
        else:
            return [
                base_lr * math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
                for base_lr in self.base_lrs
            ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class COCOPanopticEvaluatorwithVis(COCOPanopticEvaluator):
    """
    COCO Panoptic Evaluator that supports saving visualizations.
    TODO(qihangyu): Note that original implementation will also write all predictions to a tmp folder
        and then run official evaluation script, we may also check how to copy from the tmp folder for visualization.
    """

    def __init__(self, dataset_name: str, output_dir: Optional[str] = None, save_vis_num=0):
        super().__init__(dataset_name=dataset_name, output_dir=output_dir)
        self.metadata = MetadataCatalog.get("coco_2017_val_panoptic_with_sem_seg")
        self.output_dir = output_dir
        self.save_vis_num = save_vis_num

    def process(self, inputs, outputs):
        from panopticapi.utils import id2rgb

        cur_save_num = 0
        for input, output in zip(inputs, outputs):
            panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_seg = panoptic_img.cpu()
            panoptic_img = panoptic_seg.numpy()

            file_name = os.path.basename(input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            if cur_save_num < self.save_vis_num:
                image = output["original_image"]
                image = image.permute(1, 2 ,0).cpu().numpy()#[:, :, ::-1]
                visualizer = Visualizer(image, self.metadata, instance_mode=ColorMode.IMAGE)
                vis_output = visualizer.draw_panoptic_seg_predictions(
                    panoptic_seg, segments_info
                )
                if not os.path.exists(os.path.join(self.output_dir, 'vis')):
                    os.makedirs(os.path.join(self.output_dir, 'vis'))
                out_filename = os.path.join(self.output_dir, 'vis', file_name_png)
                vis_output.save(out_filename)
                cur_save_num += 1

            if segments_info is None:
                # If "segments_info" is None, we assume "panoptic_img" is a
                # H*W int32 image storing the panoptic_id in the format of
                # category_id * label_divisor + instance_id. We reserve -1 for
                # VOID label, and add 1 to panoptic_img since the official
                # evaluation script uses 0 for VOID label.
                label_divisor = self._metadata.label_divisor
                segments_info = []
                for panoptic_label in np.unique(panoptic_img):
                    if panoptic_label == -1:
                        # VOID region.
                        continue
                    pred_class = panoptic_label // label_divisor
                    isthing = (
                        pred_class in self._metadata.thing_dataset_id_to_contiguous_id.values()
                    )
                    segments_info.append(
                        {
                            "id": int(panoptic_label) + 1,
                            "category_id": int(pred_class),
                            "isthing": bool(isthing),
                        }
                    )
                # Official evaluation script uses 0 for VOID label.
                panoptic_img += 1

            
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                segments_info = [self._convert_category_id(x) for x in segments_info]
                self._predictions.append(
                    {
                        "image_id": input["image_id"],
                        "file_name": file_name_png,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                    }
                )

    def evaluate(self):
        comm.synchronize()

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            logger.info("Writing all panoptic predictions to {} ...".format(pred_dir))
            for p in self._predictions:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f)
            json_data["annotations"] = self._predictions

            output_dir = self._output_dir or pred_dir
            predictions_json = os.path.join(output_dir, "predictions.json")
            with PathManager.open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))

            from kmax_deeplab.evaluation.panoptic_evaluation import pq_compute

            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = pq_compute(
                    gt_json,
                    PathManager.get_local_path(predictions_json),
                    gt_folder=gt_folder,
                    pred_folder=pred_dir,
                )

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

        results = OrderedDict({"panoptic_seg": res})
        _print_panoptic_results(pq_res)

        return results


def _print_panoptic_results(pq_res):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    logger.info("Panoptic Evaluation Results:\n" + table)