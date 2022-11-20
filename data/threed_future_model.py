
import os
import numpy as np
from simple_3dviz import Mesh
from simple_3dviz.renderables.textured_mesh import TexturedMesh
from simple_3dviz.behaviours.misc import LightToCamera
from threed_future_labels import THREED_FUTURE_LABELS
import matplotlib.pyplot as plt
import trimesh
from mpl_toolkits.mplot3d import Axes3D
from utils import normalize_rgb, set_equal_plot_axes

try:
    from simple_3dviz.window import show
except ImportError:
    import sys
    print(
        "No GUI library found. Simple-3dviz will be running headless only.",
        file=sys.stderr
    )

class BaseThreedFutureModel(object):
    def __init__(self, model_jid, scale = 1):
        self.model_jid = model_jid
        self.scale = scale

    def _transform(self, vertices):
        # the following code is adapted and slightly simplified from the
        # 3D-Front toolbox (json2obj.py). It basically scales, rotates and
        # translates the model based on the model info.
        vertices = vertices * self.scale
        return vertices

    def mesh_renderable(
        self,
        colors=(0.5, 0.5, 0.5, 1.0),
        offset=[[0, 0, 0]],
        with_texture=False
    ):
        if  not with_texture:
            m = self.raw_model_transformed(offset)
            return Mesh.from_faces(m.vertices, m.faces, colors=colors)
        else:
            m = TexturedMesh.from_file(self.raw_model_path)
            m.scale(self.scale)
            return m


class ThreedFutureModel(BaseThreedFutureModel):
    def __init__(
        self,
        model_jid,
        model_info,
        scale,
        path_to_models
    ):
        super().__init__(model_jid, scale)
        self.model_info = model_info
        self.path_to_models = path_to_models
        self._label = None

    @property
    def raw_model_path(self):
        return os.path.join(
            self.path_to_models,
            self.model_jid,
            "raw_model.obj"
        )

    @property
    def normalized_model_path(self):
        return os.path.join(
            self.path_to_models,
            self.model_jid,
            "normalized_model.obj"
        )

    @property
    def texture_image_path(self):
        return os.path.join(
            self.path_to_models,
            self.model_jid,
            "texture.png"
        )

    def raw_model(self, skip_texture=True, skip_materials=True):
        try:
            return trimesh.load(
                self.raw_model_path,
                process=False,
                force="mesh",
                skip_materials=skip_materials,
                skip_texture=skip_texture,
            )
        except:
            import pdb
            pdb.set_trace()
            print("Loading model failed", flush=True)
            print(self.raw_model_path, flush=True)
            raise

    def raw_model_transformed(self, offset=[[0, 0, 0]]):
        model = self.raw_model()
        faces = np.array(model.faces)
        vertices = self._transform(np.array(model.vertices)) + offset

        return trimesh.Trimesh(vertices, faces)

    def centroid(self, offset=[[0, 0, 0]]):
        return self.corners(offset).mean(axis=0)

    @property
    def label(self):
        if self._label is None:
            self._label = THREED_FUTURE_LABELS[self.model_info["category"].lower()]
        return self._label

    @label.setter
    def label(self, _label):
        self._label = THREED_FUTURE_LABELS[_label.lower()]

    def show(
        self,
        behaviours=[LightToCamera()],
        offset=[[0, 0, 0]]
    ):
        renderables = self.mesh_renderable(offset=offset)
        show(renderables, behaviours=behaviours)

    def one_hot_label(self, all_labels):
        return np.eye(len(all_labels))[self.int_label(all_labels)]

    def int_label(self, all_labels):
        return all_labels.index(self.label)

    def copy_from_other_model(self, other_model):
        model = ThreedFutureModel(
            model_jid=other_model.model_jid,
            model_info=other_model.model_info,
            scale=other_model.scale,
            path_to_models=self.path_to_models
        )
        model.label = self.label
        return model

class VoxelThreedFutureModel(ThreedFutureModel):
    def __init__(
        self,
        model_jid,
        model_info,
        scale,
        path_to_models
    ):
        super().__init__(model_jid, model_info, scale, path_to_models)
        self.voxel_object = None

    def f(x):
        if x == 1:
            return [230,230,230,255]
        else:
            return [0,0,0,0]

    # Voxelize with trimesh
    def voxelize(self, pitch=0.05):
        mesh = self.raw_model(skip_texture=False, skip_materials=False)
        mesh.fill_holes()
        voxel = mesh.voxelized(pitch=pitch).hollow()
        self.voxel_object = voxel

        # Transform the texture information to color information, mapping it to each vertex. Transform it to a numpy array
        only_colors = mesh.visual.to_color().vertex_colors
        only_colors = np.asarray(only_colors)

        mesh.visual = mesh.visual.to_color()

        mesh_verts = mesh.vertices

        _,vert_idx = trimesh.proximity.ProximityQuery(mesh).vertex(voxel.points)

        cube_color=np.zeros([voxel.shape[0],voxel.shape[1],voxel.shape[2],4])

        for _, vert in enumerate(vert_idx):
            vox_verts = voxel.points_to_indices(mesh_verts[vert])
            curr_color = only_colors[vert]
            curr_color[3] = 255
            cube_color[vox_verts[0],vox_verts[1], vox_verts[2],:] = normalize_rgb(curr_color) 
        self.voxel_color_map = cube_color 
        # Fill in activated voxels with no proximity info as white voxels
        voxel_int = np.stack((voxel.matrix,)*4, axis=-1).astype(int)
        index_0 = (self.voxel_color_map == 0)
        self.voxel_color_map[index_0] = voxel_int[index_0]
        return voxel

    def get_voxel_obj_arr(self):
        if self.voxel_object == None:
            self.voxelize()
        return self.voxel_object.matrix

    def set_axes_equal(self,ax):

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    def show_voxel_plot(self, use_texture=False, preserve_axis_scale=True):
        arr = self.get_voxel_obj_arr()
        fig = plt.figure()
        ax = fig.gca(projection=Axes3D.name)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if use_texture:
            ax.voxels(arr, facecolors=self.voxel_color_map)
        else:
            ax.voxels(arr)
        if preserve_axis_scale:
            set_equal_plot_axes(ax)
        if show:
            plt.show()

    # Marching cubes reconstruction of matrix for sanity check
    def marching_cubes(self):
        voxel = self.get_voxel_obj_arr()
        mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxel, pitch=1.0)
        mesh.show()