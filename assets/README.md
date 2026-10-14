# Assets

Lookup tables the code reads through `pff.assets.asset(name)` (folder overridable with `PFF_ASSETS`). Copied from
the research assets folder; most are derived from the SMPL-X body model and fall under the
[SMPL-X licence](https://smpl-x.is.tue.mpg.de/modellicense.html); the `UV_*` files come from the DensePose release
(facebookresearch/DensePose).

| file | what | used by |
|---|---|---|
| `smplx_faces.npy` | (20908, 3) SMPL-X triangles | data, CSC, template |
| `smplx_v_template.npy` | (10475, 3) SMPL-X rest-pose vertices, metres | data (CSC target), CSC |
| `smplx_face_to_dp24.npy` | DensePose part (1..24) of every SMPL-X face | data, CSC |
| `smplx_face_uv_v2.npy` | per-face corner (U,V) on the DensePose charts, **v2**: hands and feet re-flattened without folds (the released model predicts these) | data, CSC |
| `smplx_face_uv.npy` | the same on the official **v1** charts | visualisations |
| `uv_v2_to_v1.npy` | (25, K, K, 2) v2 -> v1 chart remap for parts 3-6, so COCO (v1) can be scored | eval |
| `smplx_hidden_faces.npy` | faces never visible on a real person (eyeballs, mouth interior), dropped from supervision | data |
| `smplx_mirror.npz`, `dp24_lr_swap.npy`, `UV_symmetry_transforms.mat` | left-right mirroring of faces, parts and (U,V) | COCO flips |
| `smplx_vert_to_dp24.npy`, `smplx_vert_uv.npy`, `smplx_tangent_smooth.npy` | per-vertex part, (U,V) and a smooth tangent field | template frames, visualisations |
| `dp24_names.json`, `dp24_charts.npz`, `UV_Processed.mat` | part names, chart layout, the official DensePose chart data | visualisations |
| `mamma_holdout_seqs.txt` | MAMMA sequences held out of training (the occlusion holdout) | data, eval |

**Built, not stored:** `dp24_uv2xyz_K512.npz` (57 MB), the (part, U, V) -> template-XYZ table of the CSC loss:

```bash
python -m pff.csc --build -K 512      # a few minutes on a CPU; then: python -m pff.csc --check -K 512
```

Rebuilt from these files it is identical to the table the released model was trained with (checked).
