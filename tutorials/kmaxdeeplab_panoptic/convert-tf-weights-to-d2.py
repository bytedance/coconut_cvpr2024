import tensorflow as tf
import pickle as pkl
import sys

import torch
import numpy as np

def load_tf_weights(ckpt_path):
    # https://stackoverflow.com/questions/40118062/how-to-read-weights-saved-in-tensorflow-checkpoint-file
    from tensorflow.python.training import py_checkpoint_reader
    reader = py_checkpoint_reader.NewCheckpointReader(ckpt_path)
    state_dict = {}
    for k in reader.get_variable_to_shape_map():
        if '.OPTIMIZER_SLOT' in k or 'optimizer' in k or '_CHECKPOINTABLE_OBJECT_GRAPH' in k or 'save_counter' in k or 'global_step' in k:
            continue
        v = reader.get_tensor(k)
        state_dict[k.replace('/.ATTRIBUTES/VARIABLE_VALUE', '')] = v
    for k in sorted(state_dict.keys()):
        print(k, state_dict[k].shape)
    return state_dict

def map_bn(name1, name2):
    res = {}
    res[name1 + '/gamma'] = name2 + ".weight"
    res[name1 + '/beta'] = name2 + ".bias"
    res[name1 + '/moving_mean'] = name2 + ".running_mean"
    res[name1 + '/moving_variance'] = name2 + ".running_var"
    return res


def map_conv(name1, name2, dw=False, bias=False):
    res = {}
    if dw:
        res[name1 + '/depthwise_kernel'] = name2 + ".weight"
    else:
        res[name1 + '/kernel'] = name2 + ".weight"
    if bias:
        res[name1 + '/bias'] = name2 + ".bias"
    return res


def tf_2_torch_mapping_r50():
    res = {}
    res.update(map_conv('encoder/_stem/_conv', 'backbone.stem.conv1'))
    res.update(map_bn('encoder/_stem/_batch_norm', 'backbone.stem.conv1.norm'))
    block_num = {2: 3, 3: 4, 4: 6, 5: 3}
    for stage_idx in range(2, 6):
        for block_idx in range(1, block_num[stage_idx] + 1):
            res.update(map_conv(f'encoder/_stage{stage_idx}/_block{block_idx}/_conv1_bn_act/_conv',
             f'backbone.res{stage_idx}.{block_idx-1}.conv1'))
            res.update(map_bn(f'encoder/_stage{stage_idx}/_block{block_idx}/_conv1_bn_act/_batch_norm',
             f'backbone.res{stage_idx}.{block_idx-1}.conv1.norm'))
            res.update(map_conv(f'encoder/_stage{stage_idx}/_block{block_idx}/_conv2_bn_act/_conv',
             f'backbone.res{stage_idx}.{block_idx-1}.conv2'))
            res.update(map_bn(f'encoder/_stage{stage_idx}/_block{block_idx}/_conv2_bn_act/_batch_norm',
             f'backbone.res{stage_idx}.{block_idx-1}.conv2.norm'))
            res.update(map_conv(f'encoder/_stage{stage_idx}/_block{block_idx}/_conv3_bn/_conv',
             f'backbone.res{stage_idx}.{block_idx-1}.conv3'))
            res.update(map_bn(f'encoder/_stage{stage_idx}/_block{block_idx}/_conv3_bn/_batch_norm',
             f'backbone.res{stage_idx}.{block_idx-1}.conv3.norm'))
            res.update(map_conv(f'encoder/_stage{stage_idx}/_block{block_idx}/_shortcut/_conv',
             f'backbone.res{stage_idx}.{block_idx-1}.shortcut'))
            res.update(map_bn(f'encoder/_stage{stage_idx}/_block{block_idx}/_shortcut/_batch_norm',
             f'backbone.res{stage_idx}.{block_idx-1}.shortcut.norm'))
    return res

def tf_2_torch_mapping_convnext():
    res = {}
    for i in range(4):
        if i == 0:
            res.update(map_conv(f'encoder/downsample_layers/{i}/layer_with_weights-0',
                f'backbone.downsample_layers.{i}.0', bias=True))
            res.update(map_bn(f'encoder/downsample_layers/{i}/layer_with_weights-1',
                f'backbone.downsample_layers.{i}.1'))
        else:
            res.update(map_conv(f'encoder/downsample_layers/{i}/layer_with_weights-1',
                f'backbone.downsample_layers.{i}.1', bias=True))
            res.update(map_bn(f'encoder/downsample_layers/{i}/layer_with_weights-0',
                f'backbone.downsample_layers.{i}.0'))
    
    block_num = {0: 3, 1: 3, 2: 27, 3: 3}
    for stage_idx in range(4):
        for block_idx in range(block_num[stage_idx]):
            res.update(map_conv(f'encoder/stages/{stage_idx}/layer_with_weights-{block_idx}/depthwise_conv',
                f'backbone.stages.{stage_idx}.{block_idx}.dwconv', bias=True))
            res.update(map_bn(f'encoder/stages/{stage_idx}/layer_with_weights-{block_idx}/norm',
                f'backbone.stages.{stage_idx}.{block_idx}.norm'))
            res.update(map_conv(f'encoder/stages/{stage_idx}/layer_with_weights-{block_idx}/pointwise_conv1',
                f'backbone.stages.{stage_idx}.{block_idx}.pwconv1', bias=True))
            res.update(map_conv(f'encoder/stages/{stage_idx}/layer_with_weights-{block_idx}/pointwise_conv2',
                f'backbone.stages.{stage_idx}.{block_idx}.pwconv2', bias=True))
            res[f'encoder/stages/{stage_idx}/layer_with_weights-{block_idx}/layer_scale'] = f'backbone.stages.{stage_idx}.{block_idx}.gamma'
    
    return res

def tf_2_torch_mapping_pixel_dec():
    res = {}
    for i in range(4):
        res.update(map_bn(f'pixel_decoder/_backbone_norms/{i}', f'sem_seg_head.pixel_decoder._in_norms.{i}'))
        res.update(map_bn(f'pixel_decoder/_backbone_norms/{i}', f'sem_seg_head.pixel_decoder._in_norms.{i}'))
        res.update(map_bn(f'pixel_decoder/_backbone_norms/{i}', f'sem_seg_head.pixel_decoder._in_norms.{i}'))
        res.update(map_bn(f'pixel_decoder/_backbone_norms/{i}', f'sem_seg_head.pixel_decoder._in_norms.{i}'))
    
    for i in range(3):
        res.update(map_conv(f'pixel_decoder/_skip_connections/{i}/_resized_conv_bn1/_conv',
         f'sem_seg_head.pixel_decoder._resized_fuses.{i}._conv_bn_low.conv'))
        res.update(map_bn(f'pixel_decoder/_skip_connections/{i}/_resized_conv_bn1/_batch_norm',
         f'sem_seg_head.pixel_decoder._resized_fuses.{i}._conv_bn_low.norm'))
        res.update(map_conv(f'pixel_decoder/_skip_connections/{i}/_resized_conv_bn2/_conv',
         f'sem_seg_head.pixel_decoder._resized_fuses.{i}._conv_bn_high.conv'))
        res.update(map_bn(f'pixel_decoder/_skip_connections/{i}/_resized_conv_bn2/_batch_norm',
         f'sem_seg_head.pixel_decoder._resized_fuses.{i}._conv_bn_high.norm'))

    num_blocks = {0: 1, 1:5, 2:1, 3:1}
    for stage_idx in range(4):
        for block_idx in range(1, 1+num_blocks[stage_idx]):
            res.update(map_conv(f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_shortcut/_conv',
                f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._shortcut.conv'))
            res.update(map_bn(f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_shortcut/_batch_norm',
                f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._shortcut.norm'))
            res.update(map_conv(f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_conv1_bn_act/_conv',
                f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._conv1_bn_act.conv'))
            res.update(map_bn(f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_conv1_bn_act/_batch_norm',
                f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._conv1_bn_act.norm'))
            res.update(map_conv(f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_conv3_bn/_conv',
                f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._conv3_bn.conv'))
            res.update(map_bn(f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_conv3_bn/_batch_norm',
                f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._conv3_bn.norm'))
            if stage_idx <= 1:
                for attn in ['height', 'width']:
                    res.update(map_bn(f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_attention/_{attn}_axis/_batch_norm_qkv',
                    f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._attention._{attn}_axis._batch_norm_qkv'))
                    res.update(map_bn(f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_attention/_{attn}_axis/_batch_norm_retrieved_output',
                    f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._attention._{attn}_axis._batch_norm_retrieved_output'))
                    res.update(map_bn(f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_attention/_{attn}_axis/_batch_norm_similarity',
                    f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._attention._{attn}_axis._batch_norm_similarity'))
                    res[f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_attention/_{attn}_axis/_key_rpe/embeddings'] = (
                        f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._attention._{attn}_axis._key_rpe._embeddings.weight')
                    res[f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_attention/_{attn}_axis/_query_rpe/embeddings'] = (
                        f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._attention._{attn}_axis._query_rpe._embeddings.weight')
                    res[f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_attention/_{attn}_axis/_value_rpe/embeddings'] = (
                        f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._attention._{attn}_axis._value_rpe._embeddings.weight')
                    res[f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_attention/_{attn}_axis/qkv_kernel'] = (
                        f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._attention._{attn}_axis.qkv_transform.conv.weight')
            else:
                res.update(map_conv(f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_conv2_bn_act/_conv',
                    f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._conv2_bn_act.conv'))
                res.update(map_bn(f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_conv2_bn_act/_batch_norm',
                    f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._conv2_bn_act.norm'))
    return res     


def tf_2_torch_mapping_predcitor(prefix_tf, prefix_torch):
    res = {}
    res.update(map_bn(prefix_tf + 'pixel_space_feature_batch_norm',
        prefix_torch + '_pixel_space_head_last_convbn.norm'))
    res[prefix_tf + 'pixel_space_head/conv_block/_conv1_bn_act/_depthwise/_depthwise_conv/depthwise_kernel'] = (
        prefix_torch + '_pixel_space_head_conv0bnact.conv.weight'
    )
    res.update(map_bn(prefix_tf + 'pixel_space_head/conv_block/_conv1_bn_act/_depthwise/_batch_norm',
        prefix_torch + '_pixel_space_head_conv0bnact.norm'))
    res.update(map_conv(prefix_tf + 'pixel_space_head/conv_block/_conv1_bn_act/_pointwise/_conv',
        prefix_torch + '_pixel_space_head_conv1bnact.conv'))
    res.update(map_bn(prefix_tf + 'pixel_space_head/conv_block/_conv1_bn_act/_pointwise/_batch_norm',
        prefix_torch + '_pixel_space_head_conv1bnact.norm'))
    res.update(map_conv(prefix_tf + 'pixel_space_head/final_conv',
        prefix_torch + '_pixel_space_head_last_convbn.conv', bias=True))
    res.update(map_bn(prefix_tf + 'pixel_space_mask_batch_norm',
        prefix_torch + '_pixel_space_mask_batch_norm'))
    res.update(map_conv(prefix_tf + 'transformer_class_head/_conv',
        prefix_torch + '_transformer_class_head.conv', bias=True))
    res.update(map_conv(prefix_tf + 'transformer_mask_head/_conv',
        prefix_torch + '_transformer_mask_head.conv'))
    res.update(map_bn(prefix_tf + 'transformer_mask_head/_batch_norm',
        prefix_torch + '_transformer_mask_head.norm'))
    
    return res


def tf_2_torch_mapping_trans_dec():
    res = {}

    res.update(map_bn('transformer_decoder/_class_embedding_projection/_batch_norm',
        'sem_seg_head.predictor._class_embedding_projection.norm'))
    res.update(map_conv('transformer_decoder/_class_embedding_projection/_conv',
        'sem_seg_head.predictor._class_embedding_projection.conv'))
    res.update(map_bn('transformer_decoder/_mask_embedding_projection/_batch_norm',
        'sem_seg_head.predictor._mask_embedding_projection.norm'))
    res.update(map_conv('transformer_decoder/_mask_embedding_projection/_conv',
        'sem_seg_head.predictor._mask_embedding_projection.conv'))

    res['transformer_decoder/cluster_centers'] = 'sem_seg_head.predictor._cluster_centers.weight'

    res.update(tf_2_torch_mapping_predcitor(
            prefix_tf = '',
            prefix_torch = 'sem_seg_head.predictor._predcitor.'
        ))
    for kmax_idx in range(6):
        res.update(tf_2_torch_mapping_predcitor(
            prefix_tf = f'transformer_decoder/_kmax_decoder/{kmax_idx}/_block1_transformer/_auxiliary_clustering_predictor/_',
            prefix_torch = f'sem_seg_head.predictor._kmax_transformer_layers.{kmax_idx}._predcitor.'
        ))
        common_prefix_tf = f'transformer_decoder/_kmax_decoder/{kmax_idx}/_block1_transformer/'
        common_prefix_torch = f'sem_seg_head.predictor._kmax_transformer_layers.{kmax_idx}.'
        res.update(map_bn(common_prefix_tf + '_kmeans_memory_batch_norm_retrieved_value',
            common_prefix_torch + '_kmeans_query_batch_norm_retrieved_value'))
        res.update(map_bn(common_prefix_tf + '_kmeans_memory_conv3_bn/_batch_norm',
            common_prefix_torch + '_kmeans_query_conv3_bn.norm'))
        res.update(map_conv(common_prefix_tf + '_kmeans_memory_conv3_bn/_conv',
            common_prefix_torch + '_kmeans_query_conv3_bn.conv'))
        res.update(map_bn(common_prefix_tf + '_memory_attention/_batch_norm_retrieved_value',
            common_prefix_torch + '_query_self_attention._batch_norm_retrieved_value'))
        res.update(map_bn(common_prefix_tf + '_memory_attention/_batch_norm_similarity',
            common_prefix_torch + '_query_self_attention._batch_norm_similarity'))

        res.update(map_bn(common_prefix_tf + '_memory_conv1_bn_act/_batch_norm',
            common_prefix_torch + '_query_conv1_bn_act.norm'))
        res.update(map_conv(common_prefix_tf + '_memory_conv1_bn_act/_conv',
            common_prefix_torch + '_query_conv1_bn_act.conv'))

        res.update(map_bn(common_prefix_tf + '_memory_conv3_bn/_batch_norm',
            common_prefix_torch + '_query_conv3_bn.norm'))
        res.update(map_conv(common_prefix_tf + '_memory_conv3_bn/_conv',
            common_prefix_torch + '_query_conv3_bn.conv'))

        res.update(map_bn(common_prefix_tf + '_memory_ffn_conv1_bn_act/_batch_norm',
            common_prefix_torch + '_query_ffn_conv1_bn_act.norm'))
        res.update(map_conv(common_prefix_tf + '_memory_ffn_conv1_bn_act/_conv',
            common_prefix_torch + '_query_ffn_conv1_bn_act.conv'))

        res.update(map_bn(common_prefix_tf + '_memory_ffn_conv2_bn/_batch_norm',
            common_prefix_torch + '_query_ffn_conv2_bn.norm'))
        res.update(map_conv(common_prefix_tf + '_memory_ffn_conv2_bn/_conv',
            common_prefix_torch + '_query_ffn_conv2_bn.conv'))

        res.update(map_bn(common_prefix_tf + '_memory_qkv_conv_bn/_batch_norm',
            common_prefix_torch + '_query_qkv_conv_bn.norm'))
        res.update(map_conv(common_prefix_tf + '_memory_qkv_conv_bn/_conv',
            common_prefix_torch + '_query_qkv_conv_bn.conv'))

        res.update(map_bn(common_prefix_tf + '_pixel_conv1_bn_act/_batch_norm',
            common_prefix_torch + '_pixel_conv1_bn_act.norm'))
        res.update(map_conv(common_prefix_tf + '_pixel_conv1_bn_act/_conv',
            common_prefix_torch + '_pixel_conv1_bn_act.conv'))

        res.update(map_bn(common_prefix_tf + '_pixel_v_conv_bn/_batch_norm',
            common_prefix_torch + '_pixel_v_conv_bn.norm'))
        res.update(map_conv(common_prefix_tf + '_pixel_v_conv_bn/_conv',
            common_prefix_torch + '_pixel_v_conv_bn.conv'))

    return res


def tf_2_torch_mapping_aux_semanic_dec():
    res = {}
    res.update(map_conv('semantic_decoder/_aspp/_conv_bn_act/_conv',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._aspp._aspp_conv0.conv'))
    res.update(map_bn('semantic_decoder/_aspp/_conv_bn_act/_batch_norm',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._aspp._aspp_conv0.norm'))
    
    res.update(map_conv('semantic_decoder/_aspp/_aspp_pool/_conv_bn_act/_conv',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._aspp._aspp_pool.conv'))
    res.update(map_bn('semantic_decoder/_aspp/_aspp_pool/_conv_bn_act/_batch_norm',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._aspp._aspp_pool.norm'))

    res.update(map_conv('semantic_decoder/_aspp/_proj_conv_bn_act/_conv',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._aspp._proj_conv_bn_act.conv'))
    res.update(map_bn('semantic_decoder/_aspp/_proj_conv_bn_act/_batch_norm',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._aspp._proj_conv_bn_act.norm'))
    for i in range(1, 4):
        res.update(map_conv(f'semantic_decoder/_aspp/_aspp_conv{i}/_conv_bn_act/_conv',
             f'sem_seg_head.predictor._auxiliary_semantic_predictor._aspp._aspp_conv{i}.conv'))
        res.update(map_bn(f'semantic_decoder/_aspp/_aspp_conv{i}/_conv_bn_act/_batch_norm',
             f'sem_seg_head.predictor._auxiliary_semantic_predictor._aspp._aspp_conv{i}.norm'))
    
    res.update({
        'semantic_decoder/_fusion_conv1/_conv1_bn_act/_depthwise/_depthwise_conv/depthwise_kernel':
         'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_fusion_os8_conv0_bn_act.conv.weight'})
    res.update(map_bn('semantic_decoder/_fusion_conv1/_conv1_bn_act/_depthwise/_batch_norm',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_fusion_os8_conv0_bn_act.norm'))
    res.update({
        'semantic_decoder/_fusion_conv1/_conv1_bn_act/_pointwise/_conv/kernel':
         'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_fusion_os8_conv1_bn_act.conv.weight'})
    res.update(map_bn('semantic_decoder/_fusion_conv1/_conv1_bn_act/_pointwise/_batch_norm',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_fusion_os8_conv1_bn_act.norm'))
    
    res.update({
        'semantic_decoder/_fusion_conv2/_conv1_bn_act/_depthwise/_depthwise_conv/depthwise_kernel':
         'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_fusion_os4_conv0_bn_act.conv.weight'})
    res.update(map_bn('semantic_decoder/_fusion_conv2/_conv1_bn_act/_depthwise/_batch_norm',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_fusion_os4_conv0_bn_act.norm'))
    res.update({
        'semantic_decoder/_fusion_conv2/_conv1_bn_act/_pointwise/_conv/kernel':
         'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_fusion_os4_conv1_bn_act.conv.weight'})
    res.update(map_bn('semantic_decoder/_fusion_conv2/_conv1_bn_act/_pointwise/_batch_norm',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_fusion_os4_conv1_bn_act.norm'))

    res.update({
        'semantic_decoder/_low_level_conv1/_conv/kernel':
         'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_projection_os8.conv.weight'})
    res.update(map_bn('semantic_decoder/_low_level_conv1/_batch_norm',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_projection_os8.norm'))
    res.update({
        'semantic_decoder/_low_level_conv2/_conv/kernel':
         'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_projection_os4.conv.weight'})
    res.update(map_bn('semantic_decoder/_low_level_conv2/_batch_norm',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_projection_os4.norm'))
    

    res.update({
        'semantic_head_without_last_layer/_conv1_bn_act/_depthwise/_depthwise_conv/depthwise_kernel':
         'sem_seg_head.predictor._auxiliary_semantic_predictor.conv_block_0.conv.weight'})
    res.update(map_bn('semantic_head_without_last_layer/_conv1_bn_act/_depthwise/_batch_norm',
             'sem_seg_head.predictor._auxiliary_semantic_predictor.conv_block_0.norm'))
    res.update({
        'semantic_head_without_last_layer/_conv1_bn_act/_pointwise/_conv/kernel':
         'sem_seg_head.predictor._auxiliary_semantic_predictor.conv_block_1.conv.weight'})
    res.update(map_bn('semantic_head_without_last_layer/_conv1_bn_act/_pointwise/_batch_norm',
             'sem_seg_head.predictor._auxiliary_semantic_predictor.conv_block_1.norm'))

    res.update({
        'semantic_last_layer/kernel':
         'sem_seg_head.predictor._auxiliary_semantic_predictor.final_conv.conv.weight'})
    res.update({
        'semantic_last_layer/bias':
         'sem_seg_head.predictor._auxiliary_semantic_predictor.final_conv.conv.bias'})
    return res


# python3 convert-tf-weights-to-d2.py kmax_resnet50_coco_train/ckpt-150000 tf_kmax_r50.pkl

if __name__ == "__main__":
    input = sys.argv[1]

    state_dict = load_tf_weights(input)
    #exit()

    state_dict_torch = {}

    mapping_key = {}
    if 'resnet50' in input:
        mapping_key.update(tf_2_torch_mapping_r50())
    elif 'convnext' in input:
        mapping_key.update(tf_2_torch_mapping_convnext())
    mapping_key.update(tf_2_torch_mapping_pixel_dec())
    mapping_key.update(tf_2_torch_mapping_trans_dec())

    mapping_key.update(tf_2_torch_mapping_aux_semanic_dec())

    for k in state_dict.keys():
        value = state_dict[k]
        k2 = mapping_key[k]
        rank = len(value.shape)

        if '_batch_norm_retrieved_output' in k2 or '_batch_norm_similarity' in k2 or '_batch_norm_retrieved_value' in k2:
            value = np.reshape(value, [-1])
        elif 'qkv_transform.conv.weight' in k2:
            # (512, 1024) -> (1024, 512, 1)
            value = np.transpose(value, (1, 0))[:, :, None]
        elif '_cluster_centers.weight' in k2:
            # (1, 128, 256) -> (256, 128)
            value = np.transpose(value[0], (1, 0))
        elif '_pixel_conv1_bn_act.conv.weight' in k2:
            # (1, 512, 256) -> (256, 512, 1, 1)
            value = np.transpose(value, (2, 1, 0))[:, :, :, None]
        elif '_pixel_v_conv_bn.conv.weight' in k2:
            # (1, 256, 256) -> (256, 256, 1, 1) 
            value = np.transpose(value, (2, 1, 0))[:, :, :, None]
        elif '_pixel_space_head_conv0bnact.conv.weight' in k2:
            # (5, 5, 256, 1) -> (256, 1, 5, 5)
            value = np.transpose(value, (2, 3, 0, 1))
        elif '/layer_scale' in k:
            value = np.reshape(value, [-1])
        elif 'pwconv1.weight' in k2 or 'pwconv2.weight' in k2:
            # (128, 512) -> (512, 128)
            value = np.transpose(value, (1, 0))
        elif ('_low_level_fusion_os4_conv0_bn_act.conv.weight' in k2 
        or '_low_level_fusion_os8_conv0_bn_act.conv.weight' in k2 
        or 'sem_seg_head.predictor._auxiliary_semantic_predictor.conv_block_0.conv.weight' in k2):
            value = np.transpose(value, (2, 3, 0, 1))
        else:
            if rank == 1: # bias, norm etc
                pass
            elif rank == 2: # _query_rpe
                pass
            elif rank == 3: # conv 1d kernel, etc
                value = np.transpose(value, (2, 1, 0))
            elif rank == 4: # conv 2d kernel, etc
                value = np.transpose(value, (3, 2, 0, 1))

        state_dict_torch[k2] = value

    res = {"model": state_dict_torch, "__author__": "third_party", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)


# r50: 52.85 -> 52.71 w/ eps 1e-3
# convnext-base: 56.85 -> 56.97 w/ eps 1e-3