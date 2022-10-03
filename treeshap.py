import ctypes
import numpy as np
import glob
from shap.explainers import Tree
import pandas as pd



def custom_treeshap(model, foreground, background):
    
    # Extract tree structure with the SHAP API
    ensemble = Tree(model, data=background).model
    
    # All numpy arrays must be C_CONTIGUOUS
    assert ensemble.thresholds.flags['C_CONTIGUOUS']
    assert ensemble.features.flags['C_CONTIGUOUS']
    assert ensemble.children_left.flags['C_CONTIGUOUS']
    assert ensemble.children_right.flags['C_CONTIGUOUS']

    values = np.ascontiguousarray(ensemble.values[...,1])
    if type(foreground) == pd.DataFrame:
        foreground = np.ascontiguousarray(foreground)
    if type(background) == pd.DataFrame:
        background = np.ascontiguousarray(background)

    # Shape properties
    Nx = foreground.shape[0]
    Nz = background.shape[0]
    Nt = ensemble.features.shape[0]
    d = foreground.shape[1]
    depth = ensemble.features.shape[1]

    # Where to store the output
    results = np.zeros(foreground.shape)

    ####### Wrap C / Python #######

    # Find the shared library, the path depends on the platform and Python version
    libfile = glob.glob('build/*/treeshap*.so')[0]

    # Open the shared library
    mylib = ctypes.CDLL(libfile)

    # Tell Python the argument and result types of function main_treeshap
    mylib.main_treeshap.restype = ctypes.c_int
    mylib.main_treeshap.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                    ctypes.c_int, ctypes.c_int,
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.float64)]

    # 3. call function mysum
    mylib.main_treeshap(Nx, Nz, Nt, d, depth, foreground, background, 
                        ensemble.thresholds, values,
                        ensemble.features, ensemble.children_left, ensemble.children_right, results)

    return results



def custom_taylor_treeshap(model, foreground, background):
    
    # Extract tree structure with the SHAP API
    ensemble = Tree(model, data=background).model
    
    # All numpy arrays must be C_CONTIGUOUS
    assert ensemble.thresholds.flags['C_CONTIGUOUS']
    assert ensemble.features.flags['C_CONTIGUOUS']
    assert ensemble.children_left.flags['C_CONTIGUOUS']
    assert ensemble.children_right.flags['C_CONTIGUOUS']

    values = np.ascontiguousarray(ensemble.values[...,1])
    if type(foreground) == pd.DataFrame:
        foreground = np.ascontiguousarray(foreground).astype(np.float64)
    if type(background) == pd.DataFrame:
        background = np.ascontiguousarray(background).astype(np.float64)
    foreground = foreground.astype(np.float64)
    background = background.astype(np.float64)

    # Shape properties
    Nx = foreground.shape[0]
    Nz = background.shape[0]
    Nt = ensemble.features.shape[0]
    d = foreground.shape[1]
    depth = ensemble.features.shape[1]

    # Where to store the output
    results = np.zeros((Nx, d, d))

    ####### Wrap C / Python #######

    # Find the shared library, the path depends on the platform and Python version
    libfile = glob.glob('build/*/treeshap*.so')[0]

    # Open the shared library
    mylib = ctypes.CDLL(libfile)

    # Tell Python the argument and result types of function main_treeshap
    mylib.main_taylor_treeshap.restype = ctypes.c_int
    mylib.main_taylor_treeshap.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                    ctypes.c_int, ctypes.c_int,
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.float64)]

    # 3. call function mysum
    mylib.main_taylor_treeshap(Nx, Nz, Nt, d, depth, foreground, background, 
                                ensemble.thresholds, values,
                                ensemble.features, ensemble.children_left, ensemble.children_right, results)

    return results