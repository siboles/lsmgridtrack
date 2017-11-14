import os

def get_image_names():
    """
    Returns
    -------
    List of all available test image names
    """
    return ["reference",
            "10 percent strain"]

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
    files = {"reference": os.path.join(path, "data", "ref.nii"),
             "10 percent strain": os.path.join(path, "data", "10.nii")}
    return files[name]
