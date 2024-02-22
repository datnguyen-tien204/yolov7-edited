from typing import Optional
import subprocess
def train(weights:str,cfg:str,data:str="data/coco.yaml",hyperparameter:str='data/hyp.scratch.p5.yaml', epochs:int=300, batch_size:int=16,
          img_size=None, rect:bool=False, resume:Optional[str]=None,no_save:bool=False,no_test:bool=False,no_autoanchor:bool=False,
              evolve:bool=False,bucket:str="",cache_image:bool=False,image_weights:bool=False,device:str="", multi_scale:bool=False, single_class:bool=False,
              adam:bool=False,sync_batchnorm:bool=False,local_rank:int=-1, workers:int=8,project:str='runs/train',entity=None,name:str="Yolov7",
              exist_ok:bool=False,label_smoothing:float=0.0,quad:bool=False,linear_lr:bool=False,upload_dataset:bool=False,
              bbox_interval:int=-1,save_period:int=-1,artifact_alias:str="latest", freeze:int=0):
    if img_size is None:
        img_size = [640, 640]
        yolo_command=(f"python train.py --batch {batch_size} --cfg {cfg} --epochs {epochs} --data {data} --weights '{weights}' --hyp: {hyperparameter} --img-size {img_size} --rect {rect} --resume {resume} --nosave {no_save} "
                  f"--notest {no_test} --noautoanchor {no_autoanchor} --evolve {evolve} --bucket {bucket} --cache-images {cache_image} --image-weights {image_weights} --device {device} --multi-scale {multi_scale} --single-cls {single_class}"
                  f"--adam {adam} --sync-bn {sync_batchnorm} --local-rank {local_rank} --workers {workers} --project {project} --entity {entity} --name {name} --exist-ok {exist_ok} --quad {quad} --linear-lr {linear_lr} --label-smoothing {label_smoothing}"
                  f"--upload_dataset {upload_dataset} --bbox_interval {bbox_interval} --save_period {save_period} --artifact_alias {artifact_alias} --freeze {freeze}")

        try:
            subprocess.run(yolo_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
