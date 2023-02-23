import os


def get_image_names():
    """
    Returns
    -------
    List of all available test image names
    """
    return [
        "reference - 1 layer",
        "10 percent strain - 1 layer",
        "reference - 2 layers",
        "10 percent strain - 2 layers",
        "reference image sequence",
        "deformed image sequence",
        "2d reference image",
        "2d deformed image",
    ]


def get_image(name):
    """
    Parameters
    ----------
    name : str
        Name of the image - To see available images use :func:`get_image_names()`

    Returns
    -------
    path to requested image
    """
    path = os.path.dirname(__file__)
    files = {
        "reference - 1 layer": os.path.join(path, "data", "ref.nii"),
        "10 percent strain - 1 layer": os.path.join(path, "data", "10.nii"),
        "reference - 2 layers": os.path.join(path, "data", "ref_2layers.nii"),
        "10 percent strain - 2 layers": os.path.join(path, "data", "10_2layers.nii"),
        "reference image sequence": os.path.join(path, "data", "ref_seq"),
        "deformed image sequence": os.path.join(path, "data", "def_seq"),
        "2d reference image": os.path.join(path, "data", "ref_seq", "ref0000.tif"),
        "2d deformed image": os.path.join(path, "data", "def_seq", "def0000.tif"),
    }
    return files[name]
