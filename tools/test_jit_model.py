import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--img_width", type=int, default=16)
parser.add_argument("--img_height", type=int, default=16)
parser.add_argument("--img_channels", type=int, default=3)
parser.add_argument("--jit_checkpoint_path", type=str, required=True)
args = parser.parse_args()

example = torch.rand(args.batch_size, args.img_channels, args.img_height, args.img_width)
jit_model = torch.jit.load(args.jit_checkpoint_path)
output = jit_model(example)

print("Output shape", output.shape)
print("Output")
print(output)

