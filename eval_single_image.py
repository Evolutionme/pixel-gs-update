import argparse
import math
import numpy as np
from PIL import Image

import torch
from skimage.metrics import structural_similarity as compare_ssim


def load_image(path):
    """
    Load image as RGB float32 numpy array in [0, 1].
    """
    img = Image.open(path).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    return img


def calc_psnr(img1, img2):
    """
    img1, img2: numpy arrays in [0, 1], shape [H, W, 3]
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def calc_ssim(img1, img2):
    """
    img1, img2: numpy arrays in [0, 1], shape [H, W, 3]
    """
    return compare_ssim(
        img1,
        img2,
        channel_axis=2,
        data_range=1.0
    )


def numpy_to_lpips_tensor(img, device):
    """
    Convert numpy RGB image [H, W, 3] in [0, 1]
    to torch tensor [1, 3, H, W] in [-1, 1].
    """
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device)


def calc_lpips(img1, img2, net_type="vgg", device="cuda"):
    import lpips

    loss_fn = lpips.LPIPS(net=net_type).to(device)
    loss_fn.eval()

    t1 = numpy_to_lpips_tensor(img1, device)
    t2 = numpy_to_lpips_tensor(img2, device)

    with torch.no_grad():
        value = loss_fn(t1, t2)

    return float(value.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", type=str, required=True, help="Path to rendered image")
    parser.add_argument("--gt", type=str, required=True, help="Path to GT image")
    parser.add_argument("--lpips_net", type=str, default="vgg", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA is not available, using CPU instead.")
        args.device = "cpu"

    render = load_image(args.render)
    gt = load_image(args.gt)

    if render.shape != gt.shape:
        raise ValueError(
            f"Image size mismatch: render shape {render.shape}, GT shape {gt.shape}. "
            f"Please make sure the two images have the same resolution."
        )

    psnr_value = calc_psnr(render, gt)
    ssim_value = calc_ssim(render, gt)
    lpips_value = calc_lpips(render, gt, net_type=args.lpips_net, device=args.device)

    print("====================================")
    print("Single Image Evaluation")
    print("====================================")
    print(f"Render: {args.render}")
    print(f"GT    : {args.gt}")
    print("------------------------------------")
    print(f"SSIM :  {ssim_value:.7f}")
    print(f"PSNR : {psnr_value:.7f}")
    print(f"LPIPS:  {lpips_value:.7f}")
    print("====================================")


if __name__ == "__main__":
    main()