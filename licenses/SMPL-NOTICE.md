# SMPL-H and SMPL-X are not redistributed

`models/body_models/` is ignored by git on purpose, and no body model file is
published with this project - not in the repository, not attached to a release,
not on a model hub.

## Why

SMPL-H and SMPL-X are released by the Max Planck Institute for Intelligent
Systems under a licence for **non-commercial scientific research**, after a
registration that names the licensee. It is per-person and it forbids passing
the model files on. The same terms cover the model data, the template meshes,
the blend shapes and the joint regressors.

* SMPL-H - <https://mano.is.tue.mpg.de/> (SMPL-H is distributed with MANO)
* SMPL-X - <https://smpl-x.is.tue.mpg.de/>
* FAQ - <https://smpl.is.tue.mpg.de/modellicense.html>

## What the user does instead

1. Register on the site above with an institutional address.
2. Download `model.npz` for SMPL-H (neutral) or the SMPL-X archives.
3. Place them where the application expects them:

```text
models/body_models/smplh/neutral/model.npz
models/body_models/smplx/neutral/model.npz   (only if SMPL-X is used)
```

The application parses the `.npz` itself; nothing else is needed, and nothing
about the body model is uploaded when a project is saved.

## If you want to publish something that uses SMPL-H

Publish the *results* (the motion as BVH/FBX/glb, your code, your paper), not
the model. A generated animation is not a copy of the body model under the
licence's own reading, but a research publication still has to cite Loper et
al. 2015 / Romero et al. 2017, and commercial use needs a separate licence from
the MPI.
