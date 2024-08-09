# pip install onnx onnxsim

from argparse import ArgumentParser
from pathlib import Path

import onnx
import torch
from onnxsim import simplify
from strhub.models.utils import load_from_checkpoint


def main(source_ckpt: Path, simplify_model: bool) -> None:
    target_onnx = source_ckpt.with_suffix(".onnx")
    target_simplified_onnx = target_onnx.with_name(f"{target_onnx.stem}-sim.onnx")

    print("This script will convert the checkpoint file to ONNX.")
    print(f"Input checkpoint file: {source_ckpt}")
    print(f"Output ONNX file: {target_onnx}")
    if simplify_model:
        print(f"Output simplified ONNX file: {target_simplified_onnx}")
    print()

    print("Loading checkpoint")
    model = load_from_checkpoint(str(source_ckpt))
    # Disabling `decode_ar` so the output has a constant size.
    model.model.decode_ar = False
    # Disabling `refine_iters` because it causes errors when converting the model.
    model.model.refine_iters = 0
    model = model.eval().to("cpu")

    print("Converting to ONNX")
    dummy_input = torch.rand(1, 3, *model.hparams.img_size)
    # It's important to define the input and output names so they are constant.
    # By default they will have an auto-generated name
    model.to_onnx(
        target_onnx,
        dummy_input,
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print("Checking ONNX model")
    onnx_model = onnx.load(target_onnx)
    onnx.checker.check_model(onnx_model, full_check=True)

    if simplify_model:
        print("Simplifying")
        onnx_model_sim, check = simplify(onnx_model)
        if not check:
            print("Could not simplify the ONNX model, aborting...")
            exit(1)

        onnx.save_model(onnx_model_sim, target_simplified_onnx)

    print("Done")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Convert a PyTorch checkpoint file to ONNX."
        + "This script will output a ONNX file with the same name and location as the input checkpoint file."
        + ""
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="PyTorch checkpoint file",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--simplify",
        help="Indicates a simplified ONNX model should be generated as well. The simplified model will have the `-sim` suffix.",
        action="store_true",
        required=False,
    )
    args = parser.parse_args()

    main(args.input, args.simplify)
