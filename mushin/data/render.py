"""Scene and wrist frames at each decision, with domain randomization.

``Frames`` is an uzumaki frame hook: the world calls it with its MuJoCo model and data as the
episode runs, and ``render`` draws the latest state when the planner is asked for a decision.
Randomization touches only what the renderer reads (light colours and directions, the colours of
non-robot geoms, and the overview camera pose), and every change is undone after drawing, so the
physics and therefore the counterfactual labels are unaffected.
"""

from __future__ import annotations

import io
import random
import time
from typing import Any

import mujoco
import numpy as np
from PIL import Image

SIZE = 224
JPEG_QUALITY = 85


class Frames:
    def __init__(
        self,
        view: tuple[float, ...],
        wrist_camera: str | None,
        tool_site: str | None,
        rng: random.Random | None,
        size: int = SIZE,
    ) -> None:
        self.view = view
        self.wrist_camera = wrist_camera
        self.tool_site = tool_site
        self.rng = rng
        self.size = size
        self.model: Any = None
        self.data: Any = None
        self.renderer: Any = None
        self.render_s = 0.0

    def __call__(self, model: Any, data: Any) -> None:
        if model is not self.model and self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.model, self.data = model, data

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _scene_camera(self) -> Any:
        x, y, z, distance, azimuth, elevation = self.view
        camera = mujoco.MjvCamera()
        jitter = self.rng.uniform if self.rng else (lambda a, b: 0.0)
        camera.lookat[:] = [x + jitter(-0.02, 0.02), y + jitter(-0.02, 0.02), z]
        camera.distance = distance * (1 + jitter(-0.08, 0.08))
        camera.azimuth = azimuth + jitter(-8, 8)
        camera.elevation = elevation + jitter(-5, 5)
        return camera

    def _wrist_camera(self) -> Any:
        if self.wrist_camera is not None:
            return self.wrist_camera
        if self.tool_site is None:
            return None
        try:
            position = self.data.site(self.tool_site).xpos
        except KeyError:
            return None
        camera = mujoco.MjvCamera()
        camera.lookat[:] = position
        camera.distance = 0.4
        camera.azimuth = self.view[4]
        camera.elevation = -50.0
        return camera

    def _randomize(self) -> list[tuple[Any, Any]]:
        """Perturb render-only model fields; returns (array, saved copy) pairs to restore."""
        if self.rng is None:
            return []
        model, rng = self.model, self.rng
        saved = [(model.light_diffuse, model.light_diffuse.copy()),
                 (model.light_dir, model.light_dir.copy()),
                 (model.geom_rgba, model.geom_rgba.copy())]
        scale = rng.uniform(0.7, 1.2)
        tint = np.array([rng.uniform(0.92, 1.08) for _ in range(3)])
        model.light_diffuse[:] = np.clip(model.light_diffuse * scale * tint, 0, 1)
        model.light_dir[:] = model.light_dir + np.array(
            [[rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), 0.0] for _ in range(model.nlight)]
        )
        # Static world geoms (table, floor, walls) and free bodies (the box) get new colours;
        # robot links keep theirs.
        for geom in range(model.ngeom):
            body = model.geom_bodyid[geom]
            static = body == 0 or model.body_weldid[body] == 0
            free = model.body_jntnum[body] == 1 and model.jnt_type[model.body_jntadr[body]] == 0
            if static or free:
                rgba = model.geom_rgba[geom]
                shift = rng.uniform(-0.1, 0.1)
                rgba[:3] = np.clip(rgba[:3] + shift + [rng.uniform(-0.06, 0.06) for _ in range(3)],
                                   0, 1)
        return saved

    def render(self) -> dict[str, bytes | None]:
        if self.model is None:
            return {"scene": None, "wrist": None}
        started = time.monotonic()
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, self.size, self.size)
        saved = self._randomize()
        try:
            out: dict[str, bytes | None] = {}
            for name, camera in (("scene", self._scene_camera()), ("wrist", self._wrist_camera())):
                if camera is None:
                    out[name] = None
                    continue
                self.renderer.update_scene(self.data, camera)
                buffer = io.BytesIO()
                Image.fromarray(self.renderer.render()).save(buffer, "JPEG", quality=JPEG_QUALITY)
                out[name] = buffer.getvalue()
        finally:
            for array, copy in saved:
                array[:] = copy
        self.render_s += time.monotonic() - started
        return out
