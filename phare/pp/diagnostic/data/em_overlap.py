from phare.pp.diagnostics import Diagnostic, Patch, _EM
from .periodic_overlap import PeriodicOverlap
from .overlap import Overlap


class EMOverlap(Overlap):
    dType = _EM

    def __init__(self, patch0, patch1, dataset_key, nGhosts, offsets, sizes):
        Overlap.__init__(self, patch0, patch1, dataset_key, nGhosts, offsets, sizes)

    @classmethod
    def get(clazz, diags):
        return Overlap._get(diags, clazz)
