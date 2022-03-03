#!/usr/bin/env bash
CURDIR=$(cd $(dirname $0); pwd)
cd $CURDIR
export PYTHONPATH="$PYTHONPATH:$CURDIR"
echo 'The work dir is: ' $CURDIR

TYPE=$1
JOB_NAME=$2
ARCH=$3
KEY=$4
GPUS=$5

if [ $GPUS -lt ${MAX_GPUS:-8} ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-${MAX_GPUS:-8}}
fi

if [ -z $OUTPUT_DIR ];then
    OUTPUT_DIR=$CURDIR/work_dirs/$JOB_NAME
fi

if [ -z $PRETRAINED ];then
    PRETRAINED=$OUTPUT_DIR/checkpoint.pth
fi

# pre-training
if [[ $TYPE =~ pretrain ]]; then
    echo "==> Starting pretrainin iBOT."
    python3 -m torch.distributed.launch --nnodes ${TOTAL_NODES:-1} \
        --node_rank ${NODE_ID:-0} --nproc_per_node=$GPUS_PER_NODE \
        --master_addr=${MASTER_ADDR:-127.0.0.1} \
        --master_port=${MASTER_PORT:-29500} \
        $CURDIR/main_ibot.py \
        --arch $ARCH \
        --output_dir $OUTPUT_DIR \
        --data_path data/imagenet/train \
        ${@:6}
fi

# evaluation
if [[ $TYPE =~ imagenet_knn ]] || [[ $TYPE =~ imagenet_reg ]] || \
   [[ $TYPE =~ imagenet_linear ]] || [[ $TYPE =~ imagenet_cls ]] || \
   [[ $TYPE =~ imagenet_semi_cls ]] || [[ $TYPE =~ imagenet_unsup_cls ]] || \
   [[ $TYPE =~ cifar_cls ]] || [[ $TYPE =~ cifar10_cls ]] || [[ $TYPE =~ cars_cls ]] || \
   [[ $TYPE =~ flwrs_cls ]] || [[ $TYPE =~ inat_cls ]] || [[ $TYPE =~ inat19_cls ]] || \
   [[ $TYPE =~ coco_det ]] || [[ $TYPE =~ ade20k_dense_linear ]] || [[ $TYPE =~ ade20k_seg ]] || \
   [[ $TYPE =~ davis_viseg ]] || [[ $TYPE =~ paris_oxford_reid ]] || [[ $TYPE =~ copydays_copydet ]]; then
    
    if [ -z $AVGPOOL ] && [ -z $LINEAR_AVGPOOL ] && [ -z $LINEAR_N_LAST_BLOCKS ]; then
        if [[ $ARCH =~ small ]] || [[ $ARCH =~ swin ]]; then
            AVGPOOL=0
            LINEAR_AVGPOOL=0
            LINEAR_N_LAST_BLOCKS=4
        else
            AVGPOOL=0
            LINEAR_AVGPOOL=2
            LINEAR_N_LAST_BLOCKS=1
        fi
    fi

    echo "==> Starting evaluating iBOT."
    KEY_LIST=($(echo $KEY | tr "," "\n"))

    if [[ $TYPE =~ imagenet_knn ]]; then
        if [[ $TYPE =~ imagenet_knn ]] && [[ ! $TYPE =~ pretrain ]]; then
            for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
            do        
                python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                    --master_port=$[${MASTER_PORT:-29500}-$K] \
                    $CURDIR/evaluation/eval_knn.py \
                    --pretrained_weights $PRETRAINED \
                    --avgpool_patchtokens $AVGPOOL \
                    --arch $ARCH \
                    --checkpoint_key ${KEY_LIST[$K]} \
                    --data_path data/imagenet \
                    ${@:6}
            done
        else
            for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
            do        
                python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                    --master_port=$[${MASTER_PORT:-29500}-${K}] \
                    $CURDIR/evaluation/eval_knn.py \
                    --pretrained_weights $PRETRAINED \
                    --avgpool_patchtokens $AVGPOOL \
                    --arch $ARCH \
                    --checkpoint_key ${KEY_LIST[$K]} \
                    --data_path data/imagenet
            done
        fi
    fi
    if [[ $TYPE =~ imagenet_reg ]]; then
        if [[ $TYPE =~ imagenet_reg ]] && [[ ! $TYPE =~ pretrain ]]; then
            for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
            do        
                python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                    --master_port=$[${MASTER_PORT:-29500}-$K] \
                    $CURDIR/evaluation/eval_logistic_regression.py \
                    --pretrained_weights $PRETRAINED \
                    --avgpool_patchtokens $AVGPOOL \
                    --arch $ARCH \
                    --checkpoint_key ${KEY_LIST[$K]} \
                    --data_path data/imagenet \
                    ${@:6}
            done
        else
            for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
            do        
                python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                    --master_port=$[${MASTER_PORT:-29500}-$K] \
                    $CURDIR/evaluation/eval_logistic_regression.py \
                    --pretrained_weights $PRETRAINED \
                    --avgpool_patchtokens $AVGPOOL \
                    --arch $ARCH \
                    --checkpoint_key ${KEY_LIST[$K]} \
                    --data_path data/imagenet
            done
        fi
    fi
    if [[ $TYPE =~ imagenet_linear ]]; then    
        for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
        do  
            if [[ $TYPE =~ imagenet_linear_solo ]] && [[ ! $TYPE =~ pretrain ]]; then
                SUB_OUTPUT_DIR=$OUTPUT_DIR/linear_solo
                python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                    --master_port=$[${METIS_WORKER_0_PORT:-29500}-$K] \
                    ${CURDIR}/evaluation/eval_linear.py \
                    --pretrained_weights $PRETRAINED \
                    --n_last_blocks $LINEAR_N_LAST_BLOCKS \
                    --avgpool_patchtokens $LINEAR_AVGPOOL \
                    --arch ${ARCH} \
                    --checkpoint_key ${KEY_LIST[$K]} \
                    --output_dir $SUB_OUTPUT_DIR \
                    --data_path data/imagenet \
                    ${@:6}
            elif [[ $TYPE =~ imagenet_linear ]] && [[ ! $TYPE =~ pretrain ]]; then
                SUB_OUTPUT_DIR=$OUTPUT_DIR/linear
                python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                    --master_port=$[${MASTER_PORT:-29500}-$K] \
                    $CURDIR/evaluation/eval_linear_multi.py \
                    --pretrained_weights $PRETRAINED \
                    --n_last_blocks $LINEAR_N_LAST_BLOCKS \
                    --avgpool_patchtokens $LINEAR_AVGPOOL \
                    --arch $ARCH \
                    --checkpoint_key ${KEY_LIST[$K]} \
                    --output_dir $SUB_OUTPUT_DIR \
                    --data_path data/imagenet \
                    ${@:6}
            else
                SUB_OUTPUT_DIR=$OUTPUT_DIR/linear
                python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                    --master_port=$[${MASTER_PORT:-29500}-$K] \
                    $CURDIR/evaluation/eval_linear_multi.py \
                    --pretrained_weights $PRETRAINED \
                    --n_last_blocks $LINEAR_N_LAST_BLOCKS \
                    --avgpool_patchtokens $LINEAR_AVGPOOL \
                    --arch $ARCH \
                    --checkpoint_key ${KEY_LIST[$K]} \
                    --output_dir $SUB_OUTPUT_DIR \
                    --data_path data/imagenet
            fi
        done
    fi
    if [[ $TYPE =~ imagenet_cls ]] && [[ ! $TYPE =~ pretrain ]]; then
        SUB_OUTPUT_DIR=$OUTPUT_DIR/cls/imnet
        for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
        do
            WEIGHT_FILE=$SUB_OUTPUT_DIR/checkpoint_${KEY_LIST[$K]}.pth
            python3 $CURDIR/evaluation/classification_layer_decay/extract_backbone_weights.py \
                $PRETRAINED $WEIGHT_FILE --checkpoint_key ${KEY_LIST[$K]}
            python3 -m torch.distributed.launch --nnodes ${TOTAL_NODES:-1} \
                --node_rank ${NODE_ID:-0} --nproc_per_node=$GPUS_PER_NODE \
                --master_addr=${MASTER_ADDR:-127.0.0.1} \
                --master_port=$[${MASTER_PORT:-29500}-$K] \
                $CURDIR/evaluation/classification_layer_decay/run_class_finetuning.py \
                --finetune $WEIGHT_FILE \
                --model $ARCH \
                --epochs 100 \
                --warmup_epochs 20 \
                --layer_decay 0.65 \
                --mixup 0.8 \
                --cutmix 1.0 \
                --layer_scale_init_value 0.0 \
                --disable_rel_pos_bias \
                --abs_pos_emb \
                --use_cls \
                --imagenet_default_mean_and_std \
                --output_dir $SUB_OUTPUT_DIR \
                --data_path data/imagenet \
                ${@:6}
        done
    fi
    if [[ $TYPE =~ imagenet_semi_cls ]] && [[ ! $TYPE =~ pretrain ]]; then
        SUB_OUTPUT_DIR=${OUTPUT_DIR}/semi_cls
        for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
        do
            python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                --master_port=$[${MASTER_PORT:-29500}-$K] \
                $CURDIR/evaluation/semi_supervised/eval_cls.py \
                --pretrained_weights $PRETRAINED \
                --avgpool_patchtokens $AVGPOOL \
                --arch $ARCH \
                --checkpoint_key ${KEY_LIST[$K]} \
                --output_dir $SUB_OUTPUT_DIR \
                --data_path data/imagenet_split \
                ${@:6}
        done
    fi
    if [[ $TYPE =~ imagenet_unsup_cls ]] && [[ ! $TYPE =~ pretrain ]]; then
        for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
        do        
            python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                --master_port=$[${MASTER_PORT:-29500}-$K] \
                $CURDIR/evaluation/unsupervised/unsup_cls.py \
                --pretrained_weights $PRETRAINED \
                --arch $ARCH \
                --checkpoint_key ${KEY_LIST[$K]} \
                --data_path data/imagenet \
                ${@:6}
        done
    fi
    if [[ $TYPE =~ cifar_cls ]] && [[ ! $TYPE =~ pretrain ]]; then
        SUB_OUTPUT_DIR=$OUTPUT_DIR/cls/cifar
        for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
        do
            python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                --master_port=$[${MASTER_PORT:-29500}-$K] \
                $CURDIR/evaluation/eval_cls.py --pretrained_weights $PRETRAINED \
                --avgpool_patchtokens $AVGPOOL \
                --arch $ARCH \
                --checkpoint_key ${KEY_LIST[$K]} \
                --output_dir $SUB_OUTPUT_DIR \
                --batch-size 96 \
                --lr 7.5e-6 \
                --epochs 1000 \
                --data_set CIFAR \
                --data_path data/cifar \
                ${@:6}
        done
    fi
    if [[ $TYPE =~ cifar10_cls ]] && [[ ! $TYPE =~ pretrain ]]; then
        SUB_OUTPUT_DIR=$OUTPUT_DIR/cls/cifar10
        for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
        do
            python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                --master_port=$[${MASTER_PORT:-29500}-$K] \
                $CURDIR/evaluation/eval_cls.py \
                --pretrained_weights $PRETRAINED \
                --avgpool_patchtokens $AVGPOOL \
                --arch $ARCH \
                --checkpoint_key ${KEY_LIST[$K]} \
                --output_dir $SUB_OUTPUT_DIR \
                --batch-size 96 \
                --lr 7.5e-6 \
                --epochs 1000 \
                --data_set CIFAR10 \
                --data_path data/cifar \
                ${@:6}
        done
    fi
    if [[ $TYPE =~ cars_cls ]] && [[ ! $TYPE =~ pretrain ]]; then
        SUB_OUTPUT_DIR=$OUTPUT_DIR/cls/cars
        for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
        do
            python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                --master_port=$[${MASTER_PORT:-29500}-$K] \
                $CURDIR/evaluation/eval_cls.py --pretrained_weights $PRETRAINED \
                --avgpool_patchtokens $AVGPOOL \
                --arch $ARCH \
                --checkpoint_key ${KEY_LIST[$K]} \
                --output_dir $SUB_OUTPUT_DIR \
                --batch-size 96 \
                --lr 7.5e-5 \
                --epochs 1000 \
                --data_set Cars \
                --data_path data/cars \
                ${@:6}
        done
    fi
    if [[ $TYPE =~ flwrs_cls ]] && [[ ! $TYPE =~ pretrain ]]; then
        SUB_OUTPUT_DIR=$OUTPUT_DIR/cls/flwrs.
        for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
        do
            python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                --master_port=$[${MASTER_PORT:-29500}-$K] \
                $CURDIR/evaluation/eval_cls.py --pretrained_weights $PRETRAINED \
                --avgpool_patchtokens $AVGPOOL \
                --arch $ARCH \
                --checkpoint_key ${KEY_LIST[$K]} \
                --output_dir $SUB_OUTPUT_DIR \
                --batch-size 96 \
                --lr 7.5e-6 \
                --epochs 1000 \
                --data_set Flwrs \
                --data_path data/flwrs \
                ${@:6}
        done
    fi
    if [[ $TYPE =~ inat_cls ]] && [[ ! $TYPE =~ pretrain ]]; then
        SUB_OUTPUT_DIR=$OUTPUT_DIR/cls/inat
        for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
        do
            python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                --master_port=$[${MASTER_PORT:-29500}-$K] \
                $CURDIR/evaluation/eval_cls.py --pretrained_weights $PRETRAINED \
                --avgpool_patchtokens $AVGPOOL \
                --arch $ARCH \
                --checkpoint_key ${KEY_LIST[$K]} \
                --output_dir $SUB_OUTPUT_DIR \
                --batch-size 96 \
                --lr 7.5e-6 \
                --epochs 360 \
                --reprob 0.1 \
                --data_set INAT \
                --data_path data/inat/2018 \
                ${@:6}
        done
    fi
    if [[ $TYPE =~ inat19_cls ]] && [[ ! $TYPE =~ pretrain ]]; then
        SUB_OUTPUT_DIR=$OUTPUT_DIR/cls/inat19
        for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
        do
            python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                --master_port=$[${MASTER_PORT:-29500}-$K] \
                $CURDIR/evaluation/eval_cls.py --pretrained_weights $PRETRAINED \
                --avgpool_patchtokens $AVGPOOL \
                --arch $ARCH \
                --checkpoint_key ${KEY_LIST[$K]} \
                --output_dir $SUB_OUTPUT_DIR \
                --batch-size 96 \
                --lr 7.5e-6 \
                --epochs 360 \
                --reprob 0.1 \
                --data_set INAT19 \
                --data_path data/inat/2019 \
                ${@:6}
        done
    fi
    if [[ $TYPE =~ ade20k_dense_linear ]] && [[ ! $TYPE =~ pretrain ]]; then
        SUB_OUTPUT_DIR=$OUTPUT_DIR/seg_linear
        for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
        do
            echo "Starting evaluating ${KEY_LIST[$K]}."
            if [ -z $WEIGHT_FILE ]; then
                WEIGHT_FILE=$OUTPUT_DIR/checkpoint_${KEY_LIST[$K]}.pth
                python3 extract_backbone_weights.py $PRETRAINED $WEIGHT_FILE --checkpoint_key ${KEY_LIST[$K]}
            fi
            python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                --master_port=$[${MASTER_PORT:-29500}-$K] \
                $CURDIR/evaluation/semantic_segmentation/train.py \
                $CURDIR/evaluation/semantic_segmentation/configs/linear/${ARCH}_512_ade20k_160k.py \
                --launcher pytorch \
                --work-dir $SUB_OUTPUT_DIR \
                --deterministic \
                --options model.backbone.use_checkpoint=True \
                model.pretrained=$WEIGHT_FILE \
                ${@:6}
            python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                --master_port=$[${MASTER_PORT:-29500}-$K] \
                $CURDIR/evaluation/semantic_segmentation/test.py \
                $CURDIR/evaluation/semantic_segmentation/configs/linear/${ARCH}_512_ade20k_160k.py \
                $SUB_OUTPUT_DIR/iter_160000.pth \
                --launcher pytorch \
                --eval mIoU \
                --options model.backbone.use_checkpoint=True \
                ${@:6}
        done
    fi
    if [[ $TYPE =~ ade20k_seg ]] && [[ ! $TYPE =~ pretrain ]]; then
        SUB_OUTPUT_DIR=$OUTPUT_DIR/seg
        for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
        do
            echo "Starting evaluating ${KEY_LIST[$K]}."
            if [ -z $WEIGHT_FILE ]; then
                WEIGHT_FILE=$OUTPUT_DIR/checkpoint_${KEY_LIST[$K]}.pth
                python3 extract_backbone_weights.py $PRETRAINED $WEIGHT_FILE --checkpoint_key ${KEY_LIST[$K]}
            fi
            python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                --master_port=$[${MASTER_PORT:-29500}-$K] \
                $CURDIR/evaluation/semantic_segmentation/train.py \
                $CURDIR/evaluation/semantic_segmentation/configs/upernet/${ARCH}_512_ade20k_160k.py \
                --launcher pytorch \
                --work-dir $SUB_OUTPUT_DIR \
                --deterministic \
                --options model.backbone.use_checkpoint=True \
                model.pretrained=$WEIGHT_FILE \
                ${@:6}
            python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                --master_port=$[${MASTER_PORT:-29500}-$K] \
                $CURDIR/evaluation/semantic_segmentation/test.py \
                $CURDIR/evaluation/semantic_segmentation/configs/upernet/${ARCH}_512_ade20k_160k.py \
                $SUB_OUTPUT_DIR/iter_160000.pth \
                --launcher pytorch \
                --eval mIoU \
                --options model.backbone.use_checkpoint=True \
                ${@:6}
        done
    fi
    if [[ $TYPE =~ coco_det ]] && [[ ! $TYPE =~ pretrain ]]; then
        SUB_OUTPUT_DIR=$OUTPUT_DIR/det
        for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
        do
            echo "Starting evaluating ${KEY_LIST[$K]}."
            if [ -z $WEIGHT_FILE ]; then
                WEIGHT_FILE=$OUTPUT_DIR/checkpoint_${KEY_LIST[$K]}.pth
                python3 extract_backbone_weights.py $PRETRAINED $WEIGHT_FILE --checkpoint_key ${KEY_LIST[$K]}
            fi
            python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                --master_port=$[${MASTER_PORT:-29500}-$K] \
                $CURDIR/evaluation/object_detection/train.py \
                $CURDIR/evaluation/object_detection/configs/cascade_rcnn/${ARCH}_giou_4conv1f_coco_3x.py \
                --launcher pytorch \
                --work-dir $SUB_OUTPUT_DIR \
                --deterministic \
                --cfg-options model.backbone.use_checkpoint=True \
                model.pretrained=$WEIGHT_FILE \
                ${@:6}
            python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                --master_port=$[${MASTER_PORT:-29500}-$K] \
                $CURDIR/evaluation/object_detection/test.py \
                $CURDIR/evaluation/object_detection/configs/cascade_rcnn/${ARCH}_giou_4conv1f_coco_3x.py \
                $SUB_OUTPUT_DIR/latest.pth \
                --launcher pytorch \
                --eval bbox segm \
                --cfg-options model.backbone.use_checkpoint=True \
                ${@:6}
        done
    fi
    if [[ $TYPE =~ davis_viseg ]] && [[ ! $TYPE =~ pretrain ]]; then
        git clone git@github.com:davisvideochallenge/davis2017-evaluation.git $HOME/davis2017-evaluation
        for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
        do
            echo "Starting evaluating ${KEY_LIST[$K]}."
            python3 \
                $CURDIR/evaluation/eval_video_segmentation.py \
                --arch $ARCH \
                --pretrained_weights $PRETRAINED \
                --checkpoint_key ${KEY_LIST[$K]} \
                --data_path data/davis/DAVIS \
                --output_dir '.viseg/' \
                ${@:6}
            python3 \
                $HOME/davis2017-evaluation/evaluation_method.py \
                --task semi-supervised \
                --davis_path data/davis/DAVIS \
                --results_path '.viseg/'
        done
    fi
    if [[ $TYPE =~ paris_oxford_reid ]]; then
        for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
        do
            echo "Starting evaluating paris with ${KEY_LIST[$K]}."
            python3 -m torch.distributed.launch --use_env --nproc_per_node=1 \
                $CURDIR/evaluation/eval_image_retrieval.py \
                --avgpool_patchtokens $AVGPOOL \
                --arch $ARCH \
                --pretrained_weights $PRETRAINED \
                --checkpoint_key ${KEY_LIST[$K]} \
                --imsize 512 \
                --multiscale 1 \
                --data_path data/revisited_paris_oxford \
                --dataset rparis6k
            echo "Starting evaluating oxford with ${KEY_LIST[$K]}."
            python3 -m torch.distributed.launch --use_env --nproc_per_node=1 \
                $CURDIR/evaluation/eval_image_retrieval.py \
                --avgpool_patchtokens $AVGPOOL \
                --arch $ARCH \
                --pretrained_weights $PRETRAINED \
                --checkpoint_key ${KEY_LIST[$K]} \
                --imsize 512 \
                --multiscale 1 \
                --data_path data/revisited_paris_oxford \
                --dataset roxford5k
        done
    fi
    if [[ $TYPE =~ copydays_copydet ]]; then
        for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
        do
            echo "Starting evaluating ${KEY_LIST[$K]}."
            python3 -m torch.distributed.launch --use_env --nproc_per_node=1 \
                $CURDIR/evaluation/eval_copy_detection.py \
                --arch $ARCH \
                --pretrained_weights $PRETRAINED \
                --checkpoint_key ${KEY_LIST[$K]} \
                --data_path data/copydays \
                # --whitening_path /path/to/whitening_data \
                # --distractors_path /path/to/distractors
        done
    fi
fi

if [[ $TYPE =~ imagenet_part_linear ]]; then
    echo "==> Starting analyzing iBOT."
    KEY_LIST=($(echo ${KEY} | tr "," "\n"))
    TOPK=(1 7 14 56 112 196)
    for TK in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#TOPK[@]}-1])
    do 
        for K in $(seq ${NODE_ID:-0} ${TOTAL_NODES:-1} $[${#KEY_LIST[@]}-1]) 
        do
            python3 -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
                --master_port=$[${MASTER_PORT:-29500}-$K] \
                $CURDIR/analysis/linear_part/eval_linear_part_multi.py \
                --pretrained_weights $PRETRAINED \
                --arch $ARCH \
                --checkpoint_key ${KEY_LIST[$K]} \
                --output_dir .linear_part \
                --data_path data/imagenet \
                --topk ${TOPK[$TK]} \
                ${@:6}
        done
    done
fi