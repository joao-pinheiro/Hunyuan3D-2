# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import torch
from PIL import Image
from rembg import new_session, remove

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from hy3dgen.text2image import HunyuanDiTPipeline
import sys
import argparse
import os.path

if __name__ == '__main__':
    valid_octree = [256, 384, 512, 1024]
    model_path = 'tencent/Hunyuan3D-2'

    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--octree', type=int, default=512)
    parser.add_argument('--chunks', type=int, default=16000)
    parser.add_argument('--bgmodel', type=str, default="u2net")
    parser.add_argument('--no_texture', action='store_true')
    parser.add_argument('-i', '--input', help='Image file', required=True)
    parser.add_argument('-o', '--output', help='Generated trimesh file')
    args = parser.parse_args()

    if args.steps < 10:
        print('Error: steps should be at least 10')
        exit(-1)

    if args.octree not in [256, 384, 512, 1024]:
        print('Error: valid octree values:', ",".join(valid_octree))
        exit(-1)

    if args.chunks < 64:
        print('Error: chunks cannot be lower than 64')
        exit(-1)

    source_file = args.input
    if not os.path.isfile(source_file):
        print('Error: input file {} not found'.format(source_file))
        exit(-1)

    if args.output:
        mesh_file = args.output
        parts = os.path.basename(mesh_file).split(".")
        texture_file = os.path.join(os.path.abspath(mesh_file), parts[0] + ".texture.glb")
    else:
        parts = os.path.basename(source_file).split(".")
        mesh_file = parts[0] + ".glb"
        texture_file = parts[0] + ".texture.glb"

    if os.path.isfile(mesh_file):
        print('Error: output file {} already exists, exiting....'.format(mesh_file))
        exit(-1)

    if os.path.isfile(texture_file) and not args.no_texture:
        print('Error: output file {} already exists, exiting....'.format(mesh_file))
        exit(-1)

    # model-related code
    rembg = BackgroundRemover()
    if args.bgmodel not in rembg.model_names():
        print("Error: invalid background remover model; valid options are ", ','.join(rembg.model_names()))

    image = Image.open(source_file)
    if image.mode == 'RGB':
        session = new_session(args.bgmodel)
        image = remove(image, session=session)

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

    mesh = pipeline(image=image, num_inference_steps=args.steps, mc_algo='mc', octree_resolution=args.octree,
                    num_chunks=args.chunks,
                    generator=torch.manual_seed(2025))[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)
    mesh.export(mesh_file)

    if not args.no_texture:
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            pipeline = Hunyuan3DPaintPipeline.from_pretrained(model_path)
            mesh = pipeline(mesh, image=image)
            mesh.export(texture_file)
        except Exception as e:
            print(e)
            print('Please try to install requirements by following README.md')
