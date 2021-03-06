from ctypes import *
from numpy.ctypeslib import ndpointer
import numpy as np

lib = cdll.LoadLibrary('./bmm.dylib')

# here we CREATE A CUSTOM C TYPE
# StimMat = c_double * 12 # here you must specify the size
# with a POINTER, you can pass arrays to C, without specifying the size of the array
StimMat = POINTER(c_double) # https://stackoverflow.com/a/23248168/5623100

class Params(Structure):
    _fields_ = [("alpha", c_double),
                ("usd", c_double),
                ("JEI", c_double),
                ("T", c_double),
                ("h", c_double),
                ("cEE", c_double),
                ("cIE", c_double),
                ("cEI", c_double),
                ("cII", c_double),
                ("JEE", c_double),
                ("JEEinit", c_double),
                ("JIE", c_double),
                ("JII", c_double),
                ("JEEh", c_double),
                ("sigJ", c_double),
                ("Jtmax", c_double),
                ("Jtmin", c_double),
                ("hE", c_double),
                ("hI", c_double),
                ("IEex", c_double),
                ("IIex", c_double),
                ("mex", c_double),
                ("sigex", c_double),
                ("tmE", c_double),
                ("tmI", c_double),
                ("trec", c_double),
                ("Jepsilon", c_double),
                ("tpp", c_double),
                ("tpd", c_double),
                ("twnd", c_double),
                ("g", c_double),
                ("itauh", c_int),
                ("hsd", c_double),
                ("hh", c_double),
                ("Ip", c_double),
                ("a", c_double),
                ("xEinit", c_double),
                ("xIinit", c_double),
                ("tinit", c_double),
                ("U", c_double),
                ("taustf", c_double),
                ("taustd", c_double),
                ("Cp", c_double),
                ("Cd", c_double),
                ("HAGA", c_bool), 
                ("asym", c_bool),
                ("stimIntensity", c_double)] # https://stackoverflow.com/a/23248168/5623100] 

class retParams(Structure):
    _fields_ = [("alpha", c_double),
                ("usd", c_double),
                ("JEI", c_double),
                ("T", c_double),
                ("h", c_double),
                ("NE", c_int),
                ("NI", c_int),
                ("cEE", c_double),
                ("cIE", c_double),
                ("cEI", c_double),
                ("cII", c_double),
                ("JEE", c_double),
                ("JEEinit", c_double),
                ("JIE", c_double),
                ("JII", c_double),
                ("JEEh", c_double),
                ("sigJ", c_double),
                ("Jtmax", c_double),
                ("Jtmin", c_double),
                ("hE", c_double),
                ("hI", c_double),
                ("IEex", c_double),
                ("IIex", c_double),
                ("mex", c_double),
                ("sigex", c_double),
                ("tmE", c_double),
                ("tmI", c_double),
                ("trec", c_double),
                ("Jepsilon", c_double),
                ("tpp", c_double),
                ("tpd", c_double),
                ("twnd", c_double),
                ("g", c_double),
                ("itauh", c_int),
                ("hsd", c_double),
                ("hh", c_double),
                ("Ip", c_double),
                ("a", c_double),
                ("xEinit", c_double),
                ("xIinit", c_double),
                ("tinit", c_double),
                ("Jmin", c_double),
                ("Jmax", c_double),
                ("Cp", c_double),
                ("Cd", c_double),
                ("SNE", c_int),
                ("SNI", c_int),
                ("NEa", c_int),
                ("t", c_double),
                ("U", c_double),
                ("taustf", c_double),
                ("taustd", c_double),
                ("HAGA", c_bool), 
                ("asym", c_bool),
                ("stimIntensity", c_double)]

class cClassOne(object):
       
    # we have to specify the types of arguments and outputs of each function in the c++ class imported
    # the C types must match.

    def __init__(self, NE=2500, NI=500, cell_id=-1):

        N = NE + NI
        self.params_c_obj = Params()
        self.ret_params_c_obj = retParams()

        lib.createModel.argtypes = [c_int, c_int] # if the function gets no arguments, use None
        lib.createModel.restype = c_void_p # returns a pointer of type void

        lib.sim.argtypes = [c_void_p, c_int] # takes no args
        lib.sim.restype = c_void_p    # returns a void pointer

        lib.setParams.argtypes = [c_void_p, Structure] # takes no args
        lib.setParams.restype = c_void_p    # returns a void pointer

        lib.getState.argtypes = [c_void_p] # takes no args
        lib.getState.restype = retParams    # returns a void pointer

        lib.getWeights.argtypes = [c_void_p] # takes no args
        lib.getWeights.restype = ndpointer(dtype=c_double, ndim=2, shape=(N,N))

        lib.setWeights.argtypes = [c_void_p, ndpointer(dtype=c_double, shape=(N,N))]
        lib.setWeights.restype = c_void_p # returns a void pointer (i.e. because it doesn't have a return statement)

        lib.getF.argtypes = [c_void_p] # takes no args
        lib.getF.restype = ndpointer(dtype=np.float64, ndim=1, shape=(NE,))

        lib.getD.argtypes = [c_void_p] # takes no args
        lib.getD.restype = ndpointer(dtype=c_double, ndim=1, shape=(NE,))

        lib.getys.argtypes = [c_void_p] # takes no args
        lib.getys.restype = ndpointer(dtype=c_double, ndim=1, shape=(NE,))

        lib.setF.argtypes = [c_void_p, ndpointer(dtype=c_double, shape=(NE,))]
        lib.setF.restype = c_void_p # returns a void pointer (i.e. because it doesn't have a return statement)

        lib.setD.argtypes = [c_void_p, ndpointer(dtype=c_double, shape=(NE,))]
        lib.setD.restype = c_void_p # returns a void pointer (i.e. because it doesn't have a return statement)

        lib.setys.argtypes = [c_void_p, ndpointer(dtype=c_double, shape=(NE,))]
        lib.setys.restype = c_void_p # returns a void pointer (i.e. because it doesn't have a return statement)

        lib.setStim.argtypes = [c_void_p, ndpointer(dtype=c_int, shape=(NE,))]
        lib.setStim.restype = c_void_p # returns a void pointer (i.e. because it doesn't have a return statement)

        lib.getState.argtypes = [c_void_p] # takes no args
        lib.getState.restype = retParams    # returns a void pointer
        
        lib.dumpSpikeStates.argtypes = [c_void_p] # takes no args
        lib.dumpSpikeStates.restype = c_void_p    # returns a void pointer

        lib.loadSpikeStates.argtypes = [c_void_p, c_char_p] # c_char_p is a zero-terminated pointer to a string of characters
        lib.loadSpikeStates.restype = c_void_p    # returns a void pointer

        lib.set_t.argtypes = [c_void_p, c_double] # takes no args
        lib.set_t.restype = c_void_p    # returns a void pointer

        lib.saveSpikes.argtypes = [c_void_p, c_int] # takes no args
        lib.saveSpikes.restype = c_void_p    # returns a void pointer

        # we call the constructor from the imported libpkg.so module
        self.obj = lib.createModel(NE, NI, cell_id) # look in teh cpp code. CreateNet returns a pointer

    def setWeights(self, W):
        lib.setWeights(self.obj, W)

    # in the Python wrapper, you can name these methods anything you want. Just make sure
    # you call the right C methods (that in turn call the right C++ methods)
    
    # the order of keys defined in cluster.py IS IMPORTANT for the cClasses not to break down
    def setParams(self, params):
        for key, typ in zip(params.keys(), self.params_c_obj._fields_):
            
            # if the current field must be c_int
            if typ[1].__name__ == 'c_int':
                setattr(self.params_c_obj, key, c_int(params[key]))
            # if the current field must be c_double
            if typ[1].__name__ == 'c_double':
                setattr(self.params_c_obj, key, c_double(params[key]))
            # if the current field must be c_bool
            if typ[1].__name__ == 'c_bool':
                setattr(self.params_c_obj, key, c_bool(params[key]))
        lib.setParams(self.obj, self.params_c_obj)

    def loadSpikeStates(self, string):
        bstring = bytes(string, 'utf-8') # you must convert a python string to bytes 
        lib.loadSpikeStates(self.obj, c_char_p(bstring))

    def getState(self):
        resp = lib.getState(self.obj)
        return resp
    
    def getWeights(self):
        resp = lib.getWeights(self.obj)
        return resp
    
    def getF(self):
        resp = lib.getF(self.obj)
        return resp
    
    def setF(self, F):
        lib.setF(self.obj, F)
    
    def getD(self):
        resp = lib.getD(self.obj)
        return resp
    
    def setD(self, D):
        lib.setD(self.obj, D)
    
    def getys(self):
        resp = lib.getys(self.obj)
        return resp

    def setys(self, ys):
        lib.setys(self.obj, ys)

    def setStim(self, stimVec):
        lib.setStim(self.obj, stimVec)
    
    def sim(self, interval):
        lib.sim(self.obj, interval)

    def dumpSpikeStates(self):
        lib.dumpSpikeStates(self.obj)
    
    def set_t(self, t):
        lib.set_t(self.obj, t)

    def saveSpikes(self, saveflag):
        lib.saveSpikes(self.obj, saveflag)