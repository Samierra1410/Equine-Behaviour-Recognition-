import torch
import torch.nn as nn
import timm
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent
CKPT_PATH = BASE_DIR / "models" / "hrnet_w32_ap10k.pth"
ONNX_PATH = BASE_DIR / "ap10k_hrnet_w32.onnx"

NUM_KP     = 17
INPUT_SIZE = (256, 256)


class HRNetPose(nn.Module):
    def __init__(self, num_kp=17):
        super().__init__()
        self.hrnet = timm.create_model('hrnet_w32', pretrained=False, num_classes=0)
        self.final_layer = nn.Conv2d(32, num_kp, kernel_size=1)
        self._feat = None

        def _hook(module, inp, out):
            self._feat = out[0] if isinstance(out, (list, tuple)) else out

        self.hrnet.stage4[-1].register_forward_hook(_hook)

    def forward(self, x):
        self._feat = None
        _ = self.hrnet(x)
        assert self._feat is not None, "Hook did not fire — check HRNet stage4 structure"
        return self.final_layer(self._feat)


def remap_state_dict(mmpose_sd, model):
    our_sd  = model.state_dict()
    new_sd  = {}
    matched = 0
    skipped = []

    for mm_key, mm_val in mmpose_sd.items():
        if mm_key == 'keypoint_head.final_layer.weight':
            our_key = 'final_layer.weight'
        elif mm_key == 'keypoint_head.final_layer.bias':
            our_key = 'final_layer.bias'
        elif mm_key.startswith('backbone.'):
            our_key = 'hrnet.' + mm_key[len('backbone.'):]
        else:
            skipped.append(mm_key)
            continue

        if our_key in our_sd and our_sd[our_key].shape == mm_val.shape:
            new_sd[our_key] = mm_val
            matched += 1
        else:
            if our_key in our_sd:
                print(f"  Shape mismatch: {our_key} "
                      f"ours={tuple(our_sd[our_key].shape)} "
                      f"ckpt={tuple(mm_val.shape)}")
            skipped.append(mm_key)

    print(f"  Matched: {matched}/{len(mmpose_sd)} keys")
    if skipped:
        print(f"  Skipped: {len(skipped)} "
              f"(e.g. {skipped[:3]}{'...' if len(skipped) > 3 else ''})")

    result = our_sd.copy()
    result.update(new_sd)
    return result


def main():
    print("=" * 60)
    print(" AP-10K checkpoint -> ONNX export")
    print("=" * 60)

    if not CKPT_PATH.exists():
        print(f"\nCheckpoint not found: {CKPT_PATH}")
        print("Download hrnet_w32_ap10k_256x256.pth from the AP-10K Model Zoo")
        print("and place it at: models/hrnet_w32_ap10k.pth")
        return

    print("\n[1] Building model...")
    model = HRNetPose(NUM_KP)
    model.eval()

    print(f"\n[2] Loading checkpoint: {CKPT_PATH.name}")
    ckpt      = torch.load(str(CKPT_PATH), map_location='cpu')
    mmpose_sd = ckpt['state_dict']
    print(f"  Checkpoint keys: {len(mmpose_sd)}")

    print("\n[3] Remapping weights...")
    new_sd = remap_state_dict(mmpose_sd, model)
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"  Missing:    {len(missing)}")
    print(f"  Unexpected: {len(unexpected)}")

    print("\n[4] Verifying forward pass...")
    dummy = torch.randn(1, 3, *INPUT_SIZE)
    with torch.no_grad():
        out = model(dummy)
    print(f"  Output shape: {tuple(out.shape)}")
    print(f"  Value range:  [{out.min():.3f}, {out.max():.3f}]")
    if out.max() > 2.0:
        print("  Real trained weights confirmed.")
    else:
        print("  Warning: values unexpectedly small — check weight transfer.")

    print(f"\n[5] Exporting ONNX: {ONNX_PATH.name}")
    torch.onnx.export(
        model, dummy, str(ONNX_PATH),
        opset_version=11,
        input_names=['input'],
        output_names=['heatmaps'],
        dynamic_axes={'input': {0: 'batch'}, 'heatmaps': {0: 'batch'}},
    )
    size_mb = ONNX_PATH.stat().st_size / 1024 / 1024
    print(f"  Saved: {ONNX_PATH.name}  ({size_mb:.1f} MB)")

    print("\n[6] ONNX sanity check...")
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
    inp  = np.random.randn(1, 3, *INPUT_SIZE).astype(np.float32)
    hm   = sess.run(None, {'input': inp})[0]
    print(f"  Shape: {hm.shape}  Range: [{hm.min():.3f}, {hm.max():.3f}]")
    print(f"  ONNX working.")

    print(f"\n{'=' * 60}")
    print(f"  Done. Now run: python fig_sample_frames.py")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()