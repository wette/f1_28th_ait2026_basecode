"""
Cross-platform (Linux/Windows/macOS) offscreen OpenGL rendering to a cv2-compatible BGR image.

Dependencies:
  pip install glfw PyOpenGL numpy opencv-python shapely Pillow

Notes:
- Uses an invisible GLFW window + OpenGL framebuffer object (FBO) for offscreen rendering.
- Coordinates:
    * Ground plane is z=0
    * (0,0,0) corresponds to the TOP-LEFT of the track texture image.
    * +x to the right, +y downwards (image-like). We flip y in the projection/view convention by mapping y directly.
- Guardrails are built from Shapely LinearRing coordinates (x,y in same pixel space as track).
- Camera is vehicle-like: set_pose(x,y,z,yaw_rad). yaw=0 looks along +x direction.
- Skybox is a big cube around everything, including below the ground.

HSBIKI (Hochschule Bielefeld) – www.hsbi.de
"""

import math
import ctypes
import numpy as np
import cv2

from PIL import Image

import glfw
from OpenGL import GL as gl

try:
    from shapely.geometry import LinearRing
except ImportError:
    LinearRing = None


# -------------------------- Shader helpers --------------------------

def _compile_shader(src: str, shader_type):
    sh = gl.glCreateShader(shader_type)
    gl.glShaderSource(sh, src)
    gl.glCompileShader(sh)
    ok = gl.glGetShaderiv(sh, gl.GL_COMPILE_STATUS)
    if not ok:
        log = gl.glGetShaderInfoLog(sh).decode("utf-8", errors="replace")
        raise RuntimeError(f"Shader compile error:\n{log}\n---\n{src}")
    return sh


def _link_program(vs_src: str, fs_src: str):
    vs = _compile_shader(vs_src, gl.GL_VERTEX_SHADER)
    fs = _compile_shader(fs_src, gl.GL_FRAGMENT_SHADER)
    prog = gl.glCreateProgram()
    gl.glAttachShader(prog, vs)
    gl.glAttachShader(prog, fs)
    gl.glLinkProgram(prog)
    ok = gl.glGetProgramiv(prog, gl.GL_LINK_STATUS)
    if not ok:
        log = gl.glGetProgramInfoLog(prog).decode("utf-8", errors="replace")
        raise RuntimeError(f"Program link error:\n{log}")
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    return prog


# -------------------------- Minimal math --------------------------

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n


def _look_at(eye, target, up):
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    f = _normalize(target - eye)
    s = _normalize(np.cross(f, up))
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def _perspective(fovy_rad, aspect, znear, zfar):
    f = 1.0 / math.tan(fovy_rad / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m

# def _perspective_from_intrinsics(
#     fx,
#     fy,
#     cx,
#     cy,
#     width,
#     height,
#     znear,
#     zfar
# ):
#     """
#     Create an OpenGL projection matrix from camera intrinsics.

#     Camera coordinates:
#         x -> right
#         y -> down
#         z -> forward

#     OpenGL camera coordinates:
#         x -> right
#         y -> up
#         z -> backward
#     """

#     m = np.zeros((4, 4), dtype=np.float32)

#     m[0, 0] = 2.0 * fx / width
#     m[0, 2] = 1.0 - 2.0 * cx / width

#     m[1, 1] = -2.0 * fy / height
#     m[1, 2] = 2.0 * cy / height - 1.0

#     m[2, 2] = -(zfar + znear) / (zfar - znear)
#     m[2, 3] = -(2.0 * zfar * znear) / (zfar - znear)

#     m[3, 2] = -1.0

#     return m

def _perspective_from_intrinsics(fx, fy, cx, cy, width, height, znear, zfar):
    m = np.zeros((4, 4), dtype=np.float32)

    m[0, 0] = 2.0 * fx / width
    m[0, 2] = 2.0 * cx / width - 1.0

    m[1, 1] = 2.0 * fy / height        # - for horizontal flip
    m[1, 2] = 1.0 - 2.0 * cy / height

    m[2, 2] = -(zfar + znear) / (zfar - znear)
    m[2, 3] = -(2.0 * zfar * znear) / (zfar - znear)

    m[3, 2] = -1.0

    return m

# -------------------------- Geometry builders --------------------------

def _triangulate_poly_strip(quads):
    """
    quads: list of 4-vertex quads (each quad is [v0,v1,v2,v3], CCW)
    returns triangles vertices (v0,v1,v2) and (v0,v2,v3)
    """
    tris = []
    for q in quads:
        v0, v1, v2, v3 = q
        tris.extend([v0, v1, v2, v0, v2, v3])
    return tris


def _build_guardrail_mesh_from_linearring(linearring, z_base=10.0, height=30.0, depth=10.0, color=(0.8, 0.8, 0.8)):
    """
    Build a simple 3D ribbon-like "boxy wall" along the ring.
    We create vertical faces and a top face with a constant offset in z (depth).
    For simplicity, "depth" goes in +z direction (thickness), so it becomes a wall volume.
    """
    if linearring is None:
        raise ValueError("Shapely is not available. Please install shapely.")

    coords = list(linearring.coords)
    if len(coords) < 4:
        return None

    # Ensure closed (LinearRing should be closed already)
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    z0 = float(z_base)
    z1 = float(z_base + height)
    z2 = float(z_base + depth)
    z3 = float(z_base + depth + height)

    # Build quads for each segment: front vertical face (at z0..z1), back vertical (z2..z3), top and bottom, sides
    # We'll make a "box" extruded in +z for each segment independently (overlaps are fine for this use case).
    # Vertex format: (x,y,z, r,g,b)
    quads = []

    r, g, b = color

    for i in range(len(coords) - 1):
        (x0, y0) = coords[i]
        (x1, y1) = coords[i + 1]

        # Front face (z0..z1)
        v00 = (x0, y0, z0, r, g, b)
        v10 = (x1, y1, z0, r, g, b)
        v11 = (x1, y1, z1, r, g, b)
        v01 = (x0, y0, z1, r, g, b)
        quads.append([v00, v10, v11, v01])

        # Back face (z2..z3) - opposite winding
        w00 = (x0, y0, z2, r, g, b)
        w10 = (x1, y1, z2, r, g, b)
        w11 = (x1, y1, z3, r, g, b)
        w01 = (x0, y0, z3, r, g, b)
        quads.append([w10, w00, w01, w11])

        # Top face connecting z1 to z3
        t0 = (x0, y0, z1, r, g, b)
        t1 = (x1, y1, z1, r, g, b)
        t2 = (x1, y1, z3, r, g, b)
        t3 = (x0, y0, z3, r, g, b)
        quads.append([t0, t1, t2, t3])

        # Bottom face connecting z0 to z2
        b0 = (x0, y0, z0, r, g, b)
        b1 = (x1, y1, z0, r, g, b)
        b2 = (x1, y1, z2, r, g, b)
        b3 = (x0, y0, z2, r, g, b)
        quads.append([b1, b0, b3, b2])

    tris = _triangulate_poly_strip(quads)
    return np.array(tris, dtype=np.float32)


def _generate_skybox_pattern_texture(size=512, seed=1):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8) + 10

    # random overlapping polygons (triangles/quads)
    for _ in range(250):
        n = int(rng.integers(3, 5))
        pts = rng.integers(0, size, size=(n, 2)).astype(np.int32)
        color = rng.integers(0, 256, size=(3,), dtype=np.uint8).tolist()
        cv2.fillConvexPoly(img, pts, color)
        if rng.random() < 0.35:
            # add outline sometimes
            cv2.polylines(img, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)

    return img  # RGB-ish but OpenGL will treat as BGR? We'll upload as GL_RGB after conversion.


# -------------------------- Main class --------------------------

class openGL3d:
    def __init__(
        self,
        track_png_path="simulationRennstrecke.png",
        track_size_px=(1920, 1200),
        left_guardrail=None,   # shapely LinearRing
        right_guardrail=None,  # shapely LinearRing
        render_size=(1280, 720),
        fov_y_deg=65.0,
        fx=None,
        fy=None,
        cx=None,
        cy=None,
        distortion=None,
        near=1.0,
        far=50000.0,
        skybox_size=30000.0,
        skybox_seed=3,
    ):
        self.track_png_path = track_png_path
        self.track_w, self.track_h = track_size_px
        self.render_w, self.render_h = render_size

        self.left_guardrail = left_guardrail
        self.right_guardrail = right_guardrail

        self.fov_y = math.radians(fov_y_deg)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = distortion
        self.near = float(near)
        self.far = float(far)
        self.skybox_size = float(skybox_size)
        self.skybox_seed = int(skybox_seed)

        # Vehicle/camera pose
        self.cam_pos = np.array([100.0, 100.0, 60.0], dtype=np.float32)
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0

        # GLFW/OpenGL resources
        self._window = None
        self._fbo = None
        self._color_tex = None
        self._depth_rb = None

        self._prog_tex = None
        self._prog_col = None

        self._vao_ground = None
        self._vbo_ground = None

        self._vao_guard = None
        self._vbo_guard = None
        self._guard_vert_count = 0

        self._vao_sky = None
        self._vbo_sky = None
        self._sky_vert_count = 0

        self._tex_track = None
        self._tex_sky = None

        self._init_glfw()
        self._init_gl()
        self._build_scene()

    # ----- public API -----

    def set_pose(self, x: float, y: float, z: float, yaw_rad: float, pitch_rad: float=0.0):
        self.cam_pos = np.array([x, y, z], dtype=np.float32)
        self.cam_yaw = float(yaw_rad)
        self.cam_pitch = float(pitch_rad)


    def render(self):
        """
        Renders one frame and returns a cv2 compatible BGR image (uint8, HxWx3).
        """
        glfw.make_context_current(self._window)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)
        gl.glViewport(0, 0, self.render_w, self.render_h)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glDisable(gl.GL_BLEND)

        gl.glClearColor(0.05, 0.05, 0.08, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Matrices
        aspect = self.render_w / float(self.render_h)

        if self.fx is not None and self.fy is not None:
            proj = _perspective_from_intrinsics(
                self.fx,
                self.fy,
                self.cx,
                self.cy,
                self.render_w,
                self.render_h,
                self.near,
                self.far
            )
        else:
            proj = _perspective(self.fov_y, aspect, self.near, self.far)

        # yaw=0 => looking to +x, with y down, so "forward" in xy plane:
        fwd = np.array([
            math.cos(self.cam_yaw) * math.cos(self.cam_pitch), 
            math.sin(self.cam_yaw) * math.cos(self.cam_pitch), 
            math.sin(self.cam_pitch)
            ], dtype=np.float32
        )
        eye = self.cam_pos
        target = eye + fwd
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        view = _look_at(eye, target, up)

        # Draw skybox (textured, big cube)
        self._draw_skybox(proj, view)

        # Draw ground with track texture
        self._draw_ground(proj, view)

        # Draw guardrails (colored)
        self._draw_guardrails(proj, view)

        # Read back pixels (OpenGL gives RGB, bottom-left origin)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        data = gl.glReadPixels(0, 0, self.render_w, self.render_h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        img = np.frombuffer(data, dtype=np.uint8).reshape((self.render_h, self.render_w, 3))
        img = np.flipud(img)              # flip vertical
        img_bgr = img[:, :, ::-1].copy()  # RGB->BGR for cv2

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        return img_bgr

    def close(self):
        if self._window is None:
            return
        glfw.make_context_current(self._window)

        # Cleanup GL objects (minimal)
        for tex in [self._tex_track, self._tex_sky, self._color_tex]:
            if tex:
                gl.glDeleteTextures([tex])

        for buf in [self._vbo_ground, self._vbo_guard, self._vbo_sky]:
            if buf:
                gl.glDeleteBuffers(1, [buf])

        for vao in [self._vao_ground, self._vao_guard, self._vao_sky]:
            if vao:
                gl.glDeleteVertexArrays(1, [vao])

        for prog in [self._prog_tex, self._prog_col]:
            if prog:
                gl.glDeleteProgram(prog)

        if self._depth_rb:
            gl.glDeleteRenderbuffers(1, [self._depth_rb])
        if self._fbo:
            gl.glDeleteFramebuffers(1, [self._fbo])

        glfw.destroy_window(self._window)
        glfw.terminate()
        self._window = None

    # ----- internal setup -----

    def _init_glfw(self):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        # Invisible window for offscreen rendering; still needs a context.
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        # Try for a reasonably modern context
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        if glfw.get_platform() == glfw.PLATFORM_COCOA:
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

        self._window = glfw.create_window(64, 64, "offscreen", None, None)
        if not self._window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window/context")

        glfw.make_context_current(self._window)

    def _init_gl(self):
        # Shaders
        vs_tex = r"""
        #version 330 core
        layout(location=0) in vec3 aPos;
        layout(location=1) in vec2 aUV;
        uniform mat4 uMVP;
        out vec2 vUV;
        void main(){
            vUV = aUV;
            gl_Position = uMVP * vec4(aPos, 1.0);
        }
        """
        fs_tex = r"""
        #version 330 core
        in vec2 vUV;
        uniform sampler2D uTex;
        out vec4 FragColor;
        void main(){
            vec3 c = texture(uTex, vUV).rgb;
            FragColor = vec4(c, 1.0);
        }
        """
        self._prog_tex = _link_program(vs_tex, fs_tex)

        vs_col = r"""
        #version 330 core
        layout(location=0) in vec3 aPos;
        layout(location=1) in vec3 aCol;
        uniform mat4 uMVP;
        out vec3 vCol;
        void main(){
            vCol = aCol;
            gl_Position = uMVP * vec4(aPos, 1.0);
        }
        """
        fs_col = r"""
        #version 330 core
        in vec3 vCol;
        out vec4 FragColor;
        void main(){
            FragColor = vec4(vCol, 1.0);
        }
        """
        self._prog_col = _link_program(vs_col, fs_col)

        # Offscreen FBO
        self._fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)

        self._color_tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._color_tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, self.render_w, self.render_h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._color_tex, 0)

        self._depth_rb = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._depth_rb)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH24_STENCIL8, self.render_w, self.render_h)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_STENCIL_ATTACHMENT, gl.GL_RENDERBUFFER, self._depth_rb)

        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"FBO incomplete: {status}")

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def _build_scene(self):
        # Track texture
        self._tex_track = self._load_texture_png(self.track_png_path)

        # Ground geometry: 2 triangles, z=0, (0,0) is top-left
        # UVs: (0,0) top-left in image space, but OpenGL expects bottom-left.
        # We'll flip V to match "top-left is (0,0)" convention.
        x0, y0 = 0.0, 0.0
        x1, y1 = float(self.track_w), float(self.track_h)
        z = 0.0

        # positions + uvs
        ground = np.array([ # tri 1
            [x0, y0, z,  0.0, 1.0],
            [x1, y0, z,  1.0, 1.0],
            [x1, y1, z,  1.0, 0.0],
            # tri 2
            [x0, y0, z,  0.0, 1.0],
            [x1, y1, z,  1.0, 0.0],
            [x0, y1, z,  0.0, 0.0], ], dtype=np.float32)

        self._vao_ground = gl.glGenVertexArrays(1)
        self._vbo_ground = gl.glGenBuffers(1)
        gl.glBindVertexArray(self._vao_ground)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_ground)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, ground.nbytes, ground, gl.GL_STATIC_DRAW)

        stride = 5 * 4
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, False, stride, ctypes.c_void_p(12))
        gl.glBindVertexArray(0)

        # Guardrails
        guard_tris = []
        if self.left_guardrail is not None:
            guard_tris.append(_build_guardrail_mesh_from_linearring(self.left_guardrail))
        if self.right_guardrail is not None:
            guard_tris.append(_build_guardrail_mesh_from_linearring(self.right_guardrail))
        guard_tris = [g for g in guard_tris if g is not None and len(g) > 0]

        if guard_tris:
            guard = np.vstack(guard_tris).astype(np.float32)
            self._guard_vert_count = guard.shape[0]

            self._vao_guard = gl.glGenVertexArrays(1)
            self._vbo_guard = gl.glGenBuffers(1)

            gl.glBindVertexArray(self._vao_guard)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_guard)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, guard.nbytes, guard, gl.GL_STATIC_DRAW)

            stride = 6 * 4
            gl.glEnableVertexAttribArray(0)
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, stride, ctypes.c_void_p(0))
            gl.glEnableVertexAttribArray(1)
            gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, False, stride, ctypes.c_void_p(12))
            gl.glBindVertexArray(0)

        # Skybox: big cube with 6 faces; we map a 2D pattern texture on each face.
        self._tex_sky = self._create_texture_from_rgb(_generate_skybox_pattern_texture(512, self.skybox_seed))

        s = self.skybox_size
        # Put skybox center roughly at middle of track, also extend below ground
        cx = self.track_w * 0.5
        cy = self.track_h * 0.5
        cz = -s * 0.25  # ensure below ground as requested

        def face(a, b, c, d):
            # two triangles, with UV
            return [ (*a, 0.0, 0.0),
                (*b, 1.0, 0.0),
                (*c, 1.0, 1.0),
                (*a, 0.0, 0.0),
                (*c, 1.0, 1.0),
                (*d, 0.0, 1.0), ]

        # cube vertices
        x0, x1 = cx - s, cx + s
        y0, y1 = cy - s, cy + s
        z0, z1 = cz - s, cz + s

        # 6 faces (inside-out is not needed if we disable culling)
        sky_verts = []
        # +X
        sky_verts += face((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1))
        # -X
        sky_verts += face((x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1))
        # +Y
        sky_verts += face((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1))
        # -Y
        sky_verts += face((x1, y0, z0), (x0, y0, z0), (x0, y0, z1), (x1, y0, z1))
        # +Z (top)
        sky_verts += face((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))
        # -Z (bottom) -> explicitly below as well
        sky_verts += face((x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0))

        sky = np.array(sky_verts, dtype=np.float32)
        self._sky_vert_count = sky.shape[0]

        self._vao_sky = gl.glGenVertexArrays(1)
        self._vbo_sky = gl.glGenBuffers(1)
        gl.glBindVertexArray(self._vao_sky)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_sky)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, sky.nbytes, sky, gl.GL_STATIC_DRAW)

        stride = 5 * 4
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, False, stride, ctypes.c_void_p(12))
        gl.glBindVertexArray(0)

    # ----- draw calls -----

    def _draw_ground(self, proj, view):
        mvp = (proj @ view).astype(np.float32)

        gl.glUseProgram(self._prog_tex)
        loc = gl.glGetUniformLocation(self._prog_tex, "uMVP")
        gl.glUniformMatrix4fv(loc, 1, True, mvp)  # transpose for row-major numpy
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._tex_track)
        gl.glUniform1i(gl.glGetUniformLocation(self._prog_tex, "uTex"), 0)

        gl.glDisable(gl.GL_CULL_FACE)
        gl.glBindVertexArray(self._vao_ground)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindVertexArray(0)

    def _draw_guardrails(self, proj, view):
        if not self._vao_guard or self._guard_vert_count <= 0:
            return
        mvp = (proj @ view).astype(np.float32)

        gl.glUseProgram(self._prog_col)
        loc = gl.glGetUniformLocation(self._prog_col, "uMVP")
        gl.glUniformMatrix4fv(loc, 1, True, mvp)

        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)

        gl.glBindVertexArray(self._vao_guard)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self._guard_vert_count)
        gl.glBindVertexArray(0)

        gl.glDisable(gl.GL_CULL_FACE)

    def _draw_skybox(self, proj, view):
        # Draw skybox first; keep depth test but disable depth write for "background"
        mvp = (proj @ view).astype(np.float32)

        gl.glDepthMask(gl.GL_FALSE)
        gl.glDisable(gl.GL_CULL_FACE)

        gl.glUseProgram(self._prog_tex)
        loc = gl.glGetUniformLocation(self._prog_tex, "uMVP")
        gl.glUniformMatrix4fv(loc, 1, True, mvp)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._tex_sky)
        gl.glUniform1i(gl.glGetUniformLocation(self._prog_tex, "uTex"), 0)

        gl.glBindVertexArray(self._vao_sky)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self._sky_vert_count)
        gl.glBindVertexArray(0)

        gl.glDepthMask(gl.GL_TRUE)

    # ----- textures -----

    def _load_texture_png(self, path):
        img = Image.open(path).convert("RGBA")
        arr = np.array(img, dtype=np.uint8)

        tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        h, w = arr.shape[:2]
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, arr)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return tex

    def _create_texture_from_rgb(self, rgb_uint8):
        # rgb_uint8: HxWx3 in uint8
        tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)

        h, w = rgb_uint8.shape[:2]
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w, h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, rgb_uint8)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return tex


# -------------------------- Example main --------------------------

def main():

    #track laden
    if LinearRing is not None:
        from control.track import Track
        racetrack = Track()
        racetrack.loadFromFile("track_borders_simulation.npy")
        left = LinearRing(racetrack.innerBounds)
        right = LinearRing(racetrack.outerBounds)
    else:
        left = right = None

    left = right = None
    renderer = openGL3d(
        track_png_path="simulationRennstrecke.png",
        track_size_px=(1920, 1200),
        left_guardrail=left,
        right_guardrail=right,
        render_size=(1280, 720),
        fov_y_deg=70.0,
    )

    # Vehicle-like pose (x,y in track pixel space, z in "px" height units)
    renderer.set_pose(x=300.0, y=600.0, z=80.0, yaw_rad=0.0)

    try:
        while True:
            # Simple yaw sweep for demo
            t = glfw.get_time()
            renderer.set_pose(300.0 + 200.0 * math.cos(t * 0.2),
                              600.0 + 200.0 * math.sin(t * 0.2),
                              80.0,
                              yaw_rad=t * 0.3)

            bgr = renderer.render()
            cv2.imshow("OpenGL3D (BGR)", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or q
                break
    finally:
        renderer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()