# Security and License Review

Review scope: the minimal deliverable in this directory, compared with the user-designated source repository on 2026-11-13.

- The source repository has no top-level license file in its checked-out tree. The curvature workflow in `isaacsim/sdg/keypoint_detector/core.py` is marked Apache-2.0 and is now adapted in `src/contour_keypoints.py`; attribution and modification notes are in `NOTICE`. The separate configuration file marked `LicenseRef-NvidiaProprietary`, screenshots, extension configuration, internal namespace, and source Git history are excluded.
- Camera geometry and visualization code were independently written for this minimal demonstration. The circle, notches, and wireframe cube are synthetic. The PNG examples have no EXIF, XMP, or GPS metadata; only ordinary PNG display/color properties are present.
- Dependencies are limited to NumPy (BSD-style license) and `opencv-python-headless` (Apache-2.0 package metadata). Their wheel notices were inspected for the tested versions; this repository distributes source only and does not bundle dependency binaries. Recheck version-specific notices when updating dependencies.
- New code and synthetic examples are MIT-licensed. The adapted contour module remains under Apache-2.0 as noted in `NOTICE`; both license texts are included. No third-party image or model assets are included.
- The new Git repository must be initialized from this directory only. Do not push the source repository's history or import its branches/tags.
- This is a local pre-publication review, not legal advice or a formal third-party independent audit. The repository owner explicitly waived the separate independent review on 2026-11-13 and authorized publication. Confirm ownership and authorization to publish the underlying ideas and any claimed contribution before changing or expanding the included material.
