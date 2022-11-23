
import os
import numpy as np
from simple_3dviz import Mesh
from simple_3dviz.renderables.textured_mesh import TexturedMesh
from simple_3dviz.behaviours.misc import LightToCamera
from threed_future_labels import THREED_FUTURE_LABELS
from utils import set_equal_plot_axes, lower_slash_format, reshape_voxel_grid
import matplotlib.pyplot as plt
import trimesh
from mpl_toolkits.mplot3d import Axes3D

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
        vertices = vertices * self.scale
        return vertices
    
    def raw_model_transformed(self, offset=[[0, 0, 0]]):
        model = self.raw_model()
        faces = np.array(model.faces)
        vertices = self._transform(np.array(model.vertices)) + offset

        return trimesh.Trimesh(vertices, faces)

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

    def raw_model(self, skip_texture=False, skip_materials=False):
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

    def normalized_model(self, skip_texture=False, skip_materials=False):
        try:
            return trimesh.load(
                self.normalized_model_path,
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

    @property
    def label(self):
        if self._label is None:
            self._label = THREED_FUTURE_LABELS[lower_slash_format(self.model_info["category"])]
        return self._label

    @label.setter
    def label(self, _label):
        self._label = _label 

    def show(
        self,
        behaviours=[LightToCamera()],
        offset=[[0, 0, 0]]
    ):
        renderables = self.mesh_renderable(offset=offset)
        show(renderables, behaviours=behaviours)


class VoxelThreedFutureModel(ThreedFutureModel):
    def __init__(
        self,
        model_jid=None,
        model_info=None,
        scale=None,
        path_to_models=None,
        voxel_object=None
    ):
        self.tmesh_voxelgrid = None
        self.voxel_matrix = None
        if model_jid and model_info:
            super().__init__(model_jid, model_info, scale, path_to_models)
        if voxel_object:
            self.voxel_matrix = voxel_object["matrix"]
            self.label = voxel_object["label"]
            self.model_jid = voxel_object["model_name"]
    
    def check_voxelized(self, skip_texture=False):
        # Either voxel matrix provided directly or computed from Mesh
        if self.voxel_matrix is None:
            #If matrix not provided directly, a model_jid must be provided
            assert (self.model_jid != None), "No model to voxelize."
            self.voxelize(skip_texture=skip_texture)


    # Voxelize with trimesh. Only works if model_path provided
    def voxelize(self, pitch_factor=32, skip_texture=False):
        assert (self.model_jid != None), "No model to voxelize."
        mesh = self.normalized_model(skip_texture=skip_texture, skip_materials=skip_texture)
        #Model pitch according to longest extent
        self.tmesh_voxelgrid = mesh.voxelized(pitch=mesh.extents.max()/pitch_factor)
        sparse_indices = self.tmesh_voxelgrid.sparse_indices.T
        self.voxel_matrix = reshape_voxel_grid(sparse_indices, dims=np.array([32,32,32]), place_top=("lamp" in self.label))
        return self.tmesh_voxelgrid

    def get_voxel_matrix(self, skip_texture=False):
        self.check_voxelized(skip_texture=skip_texture)
        return self.voxel_matrix

    def get_voxel_obj_sparse(self, skip_texture=False):
        self.check_voxelized(skip_texture=skip_texture)
        return self.voxel_matrix.sparse_indices

    # Visualization of voxels on matplotlib
    def show_voxel_plot(self, use_texture=False, preserve_axis_scale=True):
        self.check_voxelized(skip_texture=use_texture)
        fig = plt.figure()
        ax = fig.gca(projection=Axes3D.name)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # Rotate axis so y points up
        ax.view_init(azim=-60, elev=120)
        if use_texture:
            ax.voxels(self.voxel_matrix, facecolors=self.voxel_color_map)
        else:
            ax.voxels(self.voxel_matrix)
        if preserve_axis_scale:
            set_equal_plot_axes(ax)
        if show:
            plt.show()

    # Alternative visualization: remeshed voxel
    def show_remeshed(self, skip_texture=False):
        self.check_voxelized(skip_texture=skip_texture)
        self.tmesh_voxelgrid.show()

    # Marching cubes reconstruction of matrix for sanity check
    def marching_cubes(self, skip_texture=False):
        self.check_voxelized(skip_texture=skip_texture)
        mesh = trimesh.voxel.ops.matrix_to_marching_cubes(self.voxel_matrix, pitch=1.0)
        mesh.show()
    
