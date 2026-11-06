#!/usr/bin/env python3
"""
ONNX → TensorRT engine converter  (run ON THE JETSON, not the server)

Converts all section_N.onnx files produced by train.py into TRT engines
that the inference_node can load.

Usage:
    python onnx_to_trt.py --onnx_dir ./models --output_dir ~/jetracer_engines
    python onnx_to_trt.py --onnx_dir ./models --output_dir ~/jetracer_engines --section 3
    python onnx_to_trt.py --onnx_dir ./models --output_dir ~/jetracer_engines --name go_stop
    python onnx_to_trt.py --onnx_dir ./models --output_dir ~/jetracer_engines --no_fp16
"""

import argparse
import os
import glob


def convert(onnx_path: str, engine_path: str, fp16: bool, workspace_mb: int):
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # parse_from_file (대신 parse(bytes)만 쓰면) external data(.onnx.data)를
    # onnx 파일과 같은 폴더에서 못 찾아서 실패한다 — 반드시 경로 기반으로 파싱.
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(f'  ONNX parse error: {parser.get_error(i)}')
        raise RuntimeError(f'Failed to parse {onnx_path}')

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)

    # train.py가 배치 축을 동적으로 export하므로(dynamic_axes), 동적 입력에는
    # optimization profile이 반드시 있어야 빌드된다. inference_node는 항상
    # 프레임 1장씩만 추론하므로 배치=1로 고정한다.
    input_tensor = network.get_input(0)
    profile = builder.create_optimization_profile()
    fixed_shape = (1,) + tuple(input_tensor.shape[1:])
    profile.set_shape(input_tensor.name, fixed_shape, fixed_shape, fixed_shape)
    config.add_optimization_profile(profile)

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print('  FP16 enabled')
        else:
            print('  FP16 not supported on this platform — using FP32')

    print(f'  Building engine (this may take a few minutes)…')
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('build_serialized_network returned None')

    with open(engine_path, 'wb') as f:
        f.write(serialized)

    print(f'  Engine saved → {engine_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Convert ONNX section models to TensorRT engines on Jetson')
    parser.add_argument('--onnx_dir', required=True,
                        help='Directory containing section_N.onnx files')
    parser.add_argument('--output_dir', required=True,
                        help='Where to write section_N.engine files')
    parser.add_argument('--section', type=int, default=None,
                        help='Convert a single section (default: convert all found)')
    parser.add_argument('--name', type=str, default=None,
                        help='Convert a single arbitrary <name>.onnx file '
                             '(e.g. --name go_stop -> go_stop.onnx -> go_stop.engine). '
                             'Takes priority over --section.')
    parser.add_argument('--no_fp16', action='store_true',
                        help='Disable FP16 mode (use FP32)')
    parser.add_argument('--workspace_mb', type=int, default=256,
                        help='TRT workspace size in MB (default: 256)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.name is not None:
        onnx_files = [os.path.join(args.onnx_dir, f'{args.name}.onnx')]
    elif args.section is not None:
        onnx_files = [os.path.join(args.onnx_dir, f'section_{args.section}.onnx')]
    else:
        onnx_files = sorted(glob.glob(os.path.join(args.onnx_dir, 'section_*.onnx')))

    if not onnx_files:
        print(f'No .onnx files found in {args.onnx_dir}')
        return

    for onnx_path in onnx_files:
        if not os.path.exists(onnx_path):
            print(f'Not found: {onnx_path}  — skipping.')
            continue
        name = os.path.splitext(os.path.basename(onnx_path))[0]
        engine_path = os.path.join(args.output_dir, f'{name}.engine')
        print(f'\nConverting {onnx_path}')
        try:
            convert(onnx_path, engine_path,
                    fp16=not args.no_fp16,
                    workspace_mb=args.workspace_mb)
        except Exception as e:
            print(f'  ERROR: {e}')

    print('\nDone.')


if __name__ == '__main__':
    main()
