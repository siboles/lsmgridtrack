import sys
from collections import MutableMapping, OrderedDict
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import numpy as np
import febio
import yaml

class FixedDict(MutableMapping):
    def __init__(self, data):
        self.__data = data

    def __len__(self):
        return len(self.__data)

    def __iter__(self):
        return iter(self.__data)

    def __setitem__(self, k, v):
        if k not in self.__data:
            raise KeyError("{:s} is not an acceptable key.".format(k))

        self.__data[k] = v

    def __delitem__(self, k):
        raise NotImplementedError

    def __getitem__(self, k):
        if k not in self.__data:
            raise KeyError("{:s} is not an acceptable key.".format(k))
        return self.__data[k]

    def __contains__(self, k):
        return k in self.__data

    def __repr__(self):
        return repr(self.__data)

class model(object):
    """
    Description
    -----------
    Create an FEBio finite element model from image and transform information.

    Parameters
    ----------
    config : str, optional
        The path to a configuration file in YAML format where analysis options are set.
        If this is not provided, some default options will be specified but others must
        be specified directly.

    options : dict, optional
        Alternatively to using a configuration file, *options* can be set by providing a
        dictionary of key, value pairs to change. Likewise, the *options* attribute of the object
        can be modified.

        .. note::
            *options* keys are immutable
        .. note::
            If both *options* and *config* are provided during object instantiation, values in *config*
            will supercede those in *options*

    Attributes
    ----------
    mesh : VTK Rectilinear Grid
        Model mesh with material properties mapped from image information.
    model : pyFEBio.model
        FEBio model defined as an XML tree

    """
    def __init__(self, **kwargs):
        # Default values
        self.options = FixedDict({
        "Image": FixedDict({
            "filename": False}),
        "Region": FixedDict({
            "origin": False,
            "x_length": False,
            "y_length": False,
            "z_length": False}),
        "Model": FixedDict({
            "transform_file": False,
            "x_edge": 1.0,
            "y_edge": 1.0,
            "z_edge": 1.0,
            "conversion_factor": 0.001,
            "isotropic": False,
            "ground_stiffness": [0.001, 1.0],
            "fibre_stiffness": [0.001, 100.0]})
    })

        self.config = None

        for key, value in kwargs.items():
            if key == "options":
                for k, v in value.items():
                    for k2, v2 in v.items():
                        self.options[k][k2] = v2
            else:
                setattr(self, key, value)

        if self.config is not None:
            self._parseConfig()


    def _parseConfig(self):
        """
        Parse configuration file in YAML format.
        """
        with open(self.config) as user:
            user_settings = yaml.load(user)

        for k, v in list(user_settings.items()):
            for k2, v2 in list(v.items()):
                self.options[k][k2] = v2


    def _readImage(self):
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(self.options["Image"]["filename"])
        reader.Update()
        self.image = reader.GetOutput()

    def _createMesh(self):
        e = 1e-14
        x_domain = np.arange(self.options["Region"]["origin"][0],
                             self.options["Region"]["origin"][0] + self.options["Region"]["x_length"] + e,
                             self.options["Model"]["x_edge"])
        y_domain = np.arange(self.options["Region"]["origin"][1],
                             self.options["Region"]["origin"][1] + self.options["Region"]["y_length"] + e,
                             self.options["Model"]["y_edge"])
        z_domain = np.arange(self.options["Region"]["origin"][2],
                             self.options["Region"]["origin"][2] + self.options["Region"]["z_length"] + e,
                             self.options["Model"]["z_edge"])

        self.mesh = vtk.vtkRectilinearGrid()
        self.mesh.SetDimensions(x_domain.size, y_domain.size, z_domain.size)
        self.mesh.SetXCoordinates(numpy_support.numpy_to_vtk(x_domain.ravel(), deep=True, array_type=vtk.VTK_FLOAT))
        self.mesh.SetYCoordinates(numpy_support.numpy_to_vtk(y_domain.ravel(), deep=True, array_type=vtk.VTK_FLOAT))
        self.mesh.SetZCoordinates(numpy_support.numpy_to_vtk(z_domain.ravel(), deep=True, array_type=vtk.VTK_FLOAT))

    def _mapProperties(self):
        probe = vtk.vtkProbeFilter()
        probe.SetSourceData(self.image)
        probe.SetInputData(self.mesh)
        probe.Update()

        p2c = vtk.vtkPointDataToCellData()
        p2c.SetInputData(probe.GetOutput())
        p2c.Update()

        result = p2c.GetOutput()

        # rescale ellipsoids to volume 1 since interpolation disrupts this
        axes = numpy_support.vtk_to_numpy(result.GetCellData().GetArray("Orientation Distribution Function"))
        shape = [i-1 for i in result.GetDimensions()][::-1] + [3,3]
        axes = axes.reshape(shape)
        lengths = np.linalg.norm(axes, axis=4)
        scale = (3.0 / (4.0 * np.pi * np.product(lengths, axis=3)))**(1./3.)
        axes *= scale[:,:,:,np.newaxis, np.newaxis]

        arr = numpy_support.numpy_to_vtk(axes.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        arr.SetNumberOfComponents(9)
        arr.SetName("Orientation Distribution Function")

        result.GetCellData().SetTensors(arr)

        intensity = numpy_support.vtk_to_numpy(result.GetCellData().GetArray("Intensity"))

        yg = self.options["Model"]["ground_stiffness"][0]
        mg = self.options["Model"]["ground_stiffness"][1] - self.options["Model"]["ground_stiffness"][0]
        ground_stiffness = yg + mg*intensity
        ground_stiffness = numpy_support.numpy_to_vtk(ground_stiffness.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        ground_stiffness.SetNumberOfComponents(1)
        ground_stiffness.SetName("Ground Modulus")

        result.GetCellData().AddArray(ground_stiffness)

        yf = self.options["Model"]["fibre_stiffness"][0]
        mf = self.options["Model"]["fibre_stiffness"][1] - self.options["Model"]["fibre_stiffness"][0]
        fibre_stiffness = yf + mf*intensity
        fibre_stiffness = numpy_support.numpy_to_vtk(fibre_stiffness.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        fibre_stiffness.SetNumberOfComponents(1)
        fibre_stiffness.SetName("Fibre Modulus")

        result.GetCellData().AddArray(fibre_stiffness)
        self.mesh = result

    def writeToVTK(self, name=None):
        """
        Description
        -----------
        Write the mesh definition with material properties mapped to a VTK rectinlinear grid file.

        Parameters
        ----------
        name : str, optional
            filename to write to (without extension). If not specified will be written as vtkmesh.vtr
        """
        if name is None:
            name = 'vtkmesh'
        writer = vtk.vtkXMLRectilinearGridWriter()
        writer.SetFileName("{:}.vtr".format(name))
        writer.SetInputData(self.mesh)
        writer.Write()

    def writeToFEBio(self, name=None):
        """
        Description
        -----------
        Write the FEBio model to file.

        Parameters
        ----------
        name : str, optional
            filename to write to (without extension). If not specified will be written as model.feb
        """
        if name is None:
            name = "model"
        m = febio.MeshDef()
        N = self.mesh.GetNumberOfPoints()
        arr = vtk.vtkFloatArray()
        arr.SetNumberOfComponents(3)
        arr.SetNumberOfTuples(N)
        points = vtk.vtkPoints()
        points.SetData(arr)
        self.mesh.GetPoints(points)
        N = self.mesh.GetNumberOfCells()
        elements = np.zeros((N, 8), int)
        for i in range(N):
            pids = vtk.vtkIdList()
            pids.SetNumberOfIds(8)
            self.mesh.GetCellPoints(i, pids)
            for j in range(8):
                elements[i,j] = pids.GetId(j) + 1
        elements = elements[:,[0,1,3,2,4,5,7,6]]
        nodes = numpy_support.vtk_to_numpy(points.GetData()) * self.options["Model"]["conversion_factor"]
        minxyz = nodes.min(axis=0) + 1e-7
        maxxyz = nodes.max(axis=0) - 1e-7
        snodes = np.where(((nodes[:,0] < minxyz[0]) | (nodes[:,0] > maxxyz[0])
                        | (nodes[:,1] < minxyz[1]) | (nodes[:,1] > maxxyz[1])
                        | (nodes[:,2] < minxyz[2]) | (nodes[:,2] > maxxyz[2])))[0]
        snodes = np.unique(snodes)
        for i, n in enumerate(nodes.tolist()):
            m.nodes.append([i+1] + n)
        for i, e in enumerate(elements.tolist()):
            m.elements.append(['hex8', i+1] + e)

        m.addElementSet(setname='cartilage',
                        eids=list(range(1,len(m.elements) +1)))
        self.model = febio.Model(modelfile="{:}.feb".format(name),
                                 steps=[{'Displace': 'solid'}])


        mats = []
        gm = numpy_support.vtk_to_numpy(self.mesh.GetCellData().GetArray('Ground Modulus'))
        gf = numpy_support.vtk_to_numpy(self.mesh.GetCellData().GetArray('Fibre Modulus'))
        odfs = numpy_support.vtk_to_numpy(self.mesh.GetCellData().GetArray('Orientation Distribution Function')).reshape(-1, 3, 3)
        if self.options["Model"]["isotropic"]:
            for i in range(3):
                for j in range(3):
                    if i==j:
                        odfs[:,i,j] = (3.0 / (np.pi * 4.0))**(1./3.)
        for i in range(gm.shape[0]):
            m.addElementSet(setname='e{:d}'.format(i+1),
                            eids=[i+1])
            mats.append(febio.MatDef(
                matid=i + 1, mname='e{:d}'.format(i+1), mtype='solid mixture',
                elsets='e{:d}'.format(i+1), attributes={
                    'mat_axis':
                    ['vector', '{:.6E},{:.6E},{:.6E}'.format(odfs[i,0,0], odfs[i,0,1], odfs[i,0,2]),
                    '{:.6E},{:.6E},{:.6E}'.format(odfs[i,1,0], odfs[i,1,1], odfs[i,1,2])]}))
            mats[-1].addBlock(branch=1, btype='solid', mtype="neo-Hookean",
                            attributes={"E": '{:.6E}'.format(gm[i]),
                                        "v": '0.15'})
            mats[-1].addBlock(branch=1, btype='solid', mtype="ellipsoidal fiber distribution",
                            attributes={"ksi": '{:.6E}, {:.6E}, {:.6E}'.format(gf[i]*np.linalg.norm(odfs[i,0,:]),
                                                                               gf[i]*np.linalg.norm(odfs[i,1,:]),
                                                                               gf[i]*np.linalg.norm(odfs[i,2,:])),
                                        "beta": '2.0, 2.0, 2.0'})
            self.model.addMaterial(mats[-1])
        self.model.addGeometry(mesh=m, mats=mats)

        ctrl = febio.Control()
        ctrl.setAttributes({'title': 'cartilage', 'max_ups': '0'})
        self.model.addControl(ctrl, step=0)

        transform = sitk.ReadTransform(self.options["Model"]["transform_file"])
        boundary = febio.Boundary(steps=1)
        for i in range(snodes.shape[0]):
            displacement = np.array(transform.TransformPoint(np.array(m.nodes[snodes[i]][1:]) * 1000.0)) / 1000.0  - np.array(m.nodes[snodes[i]][1:])
            boundary.addPrescribed(
                step=0, nodeid = snodes[i] + 1,
                dof='x', lc='1', scale=str(displacement[0]))
            boundary.addPrescribed(
                step=0, nodeid = snodes[i] + 1,
                dof='y', lc='1', scale=str(displacement[1]))
            boundary.addPrescribed(
                step=0, nodeid = snodes[i] + 1,
                dof='z', lc='1', scale=str(displacement[2]))

        self.model.addBoundary(boundary=boundary)
        self.model.addLoadCurve(lc='1', lctype='linear', points=[0, 0, 1, 1])

        self.model.writeModel()

    def generate(self):
        """
        Description
        -----------
        Generate the model.
        """
        for k, v in self.options.items():
            for k2, v2 in v.items():
                if not v2 and k2 != "isotropic":
                    raise ValueError("Option, {:s}, in section, {:s}, must be specified to generate the model.".format(k2, k))
        self._readImage()
        self._createMesh()
        self._mapProperties()
