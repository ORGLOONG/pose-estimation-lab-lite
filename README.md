# Pose Estimation Lab (Minimal)

A small, standalone demonstration of contour curvature, geometric keypoint selection, rigid camera transforms, and pinhole projection. The examples are synthetic; no simulator, internal SDK, private model, or customer data is needed.

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/visualization.py
```

The last command regenerates `examples/sample_input.png` and `examples/sample_output.png`. `sample_input.png` is a synthetic circular mask with shallow notches; the output overlays selected contour keypoints and a projected wireframe cube.

## Modules

- `src/contour_keypoints.py`: external contour extraction, local turning-angle curvature, fixed/adaptive/multi-level selection, and spatial suppression. The selection workflow is adapted from the Apache-2.0 detector noted in `NOTICE`.
- `src/camera_geometry.py`: rigid 4x4 transforms, inverse transforms, focal-length calibration from vertical field of view, and pinhole projection. Camera coordinates use +X right, +Y down, +Z forward.
- `src/visualization.py`: deterministic synthetic examples.

## Coordinate convention

Points are row-wise XYZ in camera coordinates and distances use consistent arbitrary units (meters are conventional). Intrinsics are in pixels. Points with non-positive camera Z are returned as NaN because they are not in front of the camera.

## License

The original code and synthetic examples in this repository are released under the MIT License (see `LICENSE`).
