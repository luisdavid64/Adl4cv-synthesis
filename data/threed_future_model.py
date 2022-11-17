
import os
import numpy as np
from simple_3dviz import Mesh
from simple_3dviz.renderables.textured_mesh import TexturedMesh
from simple_3dviz.behaviours.misc import LightToCamera
from threed_future_labels import THREED_FUTURE_LABELS
import matplotlib.pyplot as plt
import trimesh

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

    # Voxelize with trimesh
    def voxelize(self, pitch=0.01):
        mesh = self.raw_model(skip_texture=False, skip_materials=False)
        mesh.fill_holes()
        voxel = mesh.voxelized(pitch=pitch).hollow()
        self.voxel_object = voxel

        # Transform the texture information to color information, mapping it to each vertex. Transform it to a numpy array
        only_colors = mesh.visual.to_color().vertex_colors
        only_colors = np.asarray(only_colors)
        # If we want to add the color information to the mesh uncomment this part
        mesh.visual = mesh.visual.to_color()

        # Extract the mesh vertices
        mesh_verts = mesh.vertices

        # We use the ProximityQuery built-in function to get the closest voxel point centers to each vertex of the mesh
        _,vert_idx = trimesh.proximity.ProximityQuery(mesh).vertex(voxel.points)

        # We initialize a array of zeros of size X,Y,Z,4 to contain the colors for each voxel of the voxelized mesh in the grid
        cube_color=np.zeros([voxel.shape[0],voxel.shape[1],voxel.shape[2],4])

        # We loop through all the calculated closest voxel points
        for _, vert in enumerate(vert_idx):
            # Get the voxel grid index of each closets voxel center point
            vox_verts = voxel.points_to_indices(mesh_verts[vert])
            # Get the color vertex color
            curr_color = only_colors[vert]
            # Set the alpha channel of the color
            curr_color[3] = 255
            # add the color to the specific voxel grid index 
            cube_color[vox_verts[0],vox_verts[1], vox_verts[2],:] = curr_color
        # generate a voxelized mesh from the voxel grid representation, using the calculated colors 
        return voxel

    def get_voxel_obj_arr(self):
        if self.voxel_object == None:
            self.voxelize()
        return self.voxel_object.matrix
    
    def show_voxel_plot(self):
        arr = self.get_voxel_obj_arr()
        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(arr)
        plt.show()