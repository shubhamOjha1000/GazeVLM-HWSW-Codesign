import cv2, numpy as np, torch, torch.nn.functional as F

_DINO = None
def get_dino(device):
    global _DINO
    if _DINO is None:
        _DINO = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()
        for p in _DINO.parameters(): p.requires_grad_(False)
    return _DINO

@torch.no_grad()
def frame_features(frame_rgb, device, dino_size=98):
    """SINGLE DINOv2 pass -> (cls_token, patch_tokens_grid, grid)."""
    m = get_dino(device)
    t = torch.from_numpy(frame_rgb.copy()).permute(2,0,1).float().unsqueeze(0).to(device)/255.0
    t = F.interpolate(t, size=(dino_size, dino_size), mode='bilinear', align_corners=False)
    out = m.forward_features(t)
    cls = F.normalize(out["x_norm_clstoken"][0], dim=-1)
    patches = F.normalize(out["x_norm_patchtokens"][0], dim=-1)
    grid = int(round(patches.shape[0] ** 0.5))
    return cls, patches, grid

def patch_token_at_gaze(patches, grid, gaze_xy_norm):
    gx = min(grid-1, int(np.clip(gaze_xy_norm[0], 0, 1) * grid))
    gy = min(grid-1, int(np.clip(gaze_xy_norm[1], 0, 1) * grid))
    return patches[gy*grid + gx]

def cosine(a, b):
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

def gaze_crop_rgb(frame, g, patch_px=64):
    g = np.where(np.isfinite(g), g, 0.5)
    h, w = frame.shape[:2]; cx, cy = g[0]*w, g[1]*h; half = patch_px//2
    x0,x1 = int(np.clip(cx-half,0,w-1)), int(np.clip(cx+half,0,w))
    y0,y1 = int(np.clip(cy-half,0,h-1)), int(np.clip(cy+half,0,h))
    c = frame[y0:y1, x0:x1]
    if c.shape[:2] != (patch_px, patch_px): c = cv2.resize(c, (patch_px, patch_px))
    return c

def show_pair(frame_a, frame_b, gaze_a, gaze_b, frame_sim, patch_sim,
              patch_px=64, save_path=None, grid=None):
    """Save (and try to display) the two frames + gaze markers + gaze patches, annotated
    with the two similarities. Also draws the selected patch cell so you can confirm the
    DINOv2 gaze-patch token overlaps the gaze marker."""
    import matplotlib
    matplotlib.use("Agg", force=True)   # file-safe backend; works headless / as a script
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    crop_a = gaze_crop_rgb(frame_a, gaze_a, patch_px)
    crop_b = gaze_crop_rgb(frame_b, gaze_b, patch_px)
    fig, ax = plt.subplots(2, 2, figsize=(9, 9))
    for a, img, g, t in [(ax[0,0], frame_a, gaze_a, "frame t-1"),
                         (ax[0,1], frame_b, gaze_b, "frame t")]:
        h, w = img.shape[:2]
        a.imshow(img)
        a.scatter([g[0]*w], [g[1]*h], s=160, ec='yellow', fc='none', lw=2)
        if grid:  # draw the DINOv2 patch cell used for the gaze-patch token
            gx = min(grid-1, int(np.clip(g[0],0,1)*grid)); gy = min(grid-1, int(np.clip(g[1],0,1)*grid))
            a.add_patch(mpatches.Rectangle((gx*w/grid, gy*h/grid), w/grid, h/grid,
                                           fill=False, ec='cyan', lw=2))
        a.set_title(t); a.axis('off')
    ax[1,0].imshow(crop_a); ax[1,0].set_title("gaze patch t-1"); ax[1,0].axis('off')
    ax[1,1].imshow(crop_b); ax[1,1].set_title("gaze patch t"); ax[1,1].axis('off')
    fig.suptitle(f"frame_sim={frame_sim:.3f}   gaze_patch_token_sim={patch_sim:.3f}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"  [viz] saved {save_path}")
    plt.close(fig)