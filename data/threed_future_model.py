
import os
import numpy as np
from simple_3dviz import Mesh
from simple_3dviz.renderables.textured_mesh import TexturedMesh
from simple_3dviz.behaviours.misc import LightToCamera
from threed_future_labels import THREED_FUTURE_LABELS
from utils import set_equal_plot_axes, lower_slash_format
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
        # the following code is adapted and slightly simplified from the
        # 3D-Front toolbox (json2obj.py). It basically scales, rotates and
        # translates the model based on the model info.
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
        self._label = THREED_FUTURE_LABELS[lower_slash_format(_label)]

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
        model_jid=None,
        model_info=None,
        scale=None,
        path_to_models=None,
        voxel_object=None
    ):
        self.voxel_object = None
        if model_jid:
            super().__init__(model_jid, model_info, scale, path_to_models)
            self.voxel_object = None
        if voxel_object:
            self.voxel_object = voxel_object
            self.label = voxel_object["label"]
            self.model_jid = voxel_object["model_name"]
    
    # Labels should now be structured
    @property
    def label(self):
        return self._label
        
    @label.setter
    def label(self, _label):
        self._label = _label

    # Voxelize with trimesh. Only works if model_path provided
    def voxelize(self, pitch=1/31, skip_texture=False):
        if self.model_jid:
            mesh = self.normalized_model(skip_texture=skip_texture, skip_materials=skip_texture)
            mesh.fill_holes()
            # Pack mesh into unit cube to normalize voxel creation
            dim_scales = mesh.extents
            mesh.apply_scale(1.0 / dim_scales)
            self.voxel_object = mesh.voxelized(pitch=pitch).hollow()

            #Re-scale voxel dimensions with transform
            re_scale = np.diag((np.append(dim_scales,1)))
            self.voxel_object.apply_transform(re_scale)
            return self.voxel_object
        else:
            print("Error: No model to voxelize")

    def get_voxel_obj_matrix(self, skip_texture=False):
        if self.voxel_object == None:
            self.voxelize(skip_texture=skip_texture)
        return self.voxel_object["matrix"]

    # Visualization of voxels on matplotlib
    def show_voxel_plot(self, use_texture=False, preserve_axis_scale=True):
        arr = self.get_voxel_obj_matrix()
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

    # Alternative visualization: remeshed voxel
    def show_remeshed(self):
        if self.voxel_object == None:
            self.voxelize()
        self.voxel_object.show()

    # Marching cubes reconstruction of matrix for sanity check
    def marching_cubes(self):
        voxel = self.get_voxel_obj_matrix()
        mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxel, pitch=1.0)
        mesh.show()
    
