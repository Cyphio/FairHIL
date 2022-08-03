from bokeh.io import *
from bokeh.models import *
from bokeh.layouts import *
from bokeh.plotting import *


class FairHIL:

    def __init__(self, DATASET, MODE, SENSI_FEATS, TARGET_FEAT, DEEP_DIVE_METRICS, PRIMARY_METRIC, BINNING, CAUSAL_ALGO, CARD_GEN):
        self.DATASET = DATASET
        self.MODE = MODE
        self.SENSI_FEATS = SENSI_FEATS
        self.TARGET_FEAT = TARGET_FEAT
        self.DEEP_DIVE_METRICS = DEEP_DIVE_METRICS
        self.PRIMARY_METRIC = PRIMARY_METRIC
        self.BINNING = BINNING
        self.CAUSAL_ALGO = CAUSAL_ALGO
        self.CARD_GEN = CARD_GEN

        self.causal_graph = CausalGraphView()
        self.overview = SystemOverview()
        self.relationships = RelationshipsView()
        self.explore_dataset = DatasetView()
        self.combinations = CombinationView()
        self.comparator = ComparisonView()

    def launch_self(self):
        curdoc().title = "FairHIL"
        curdoc().clear()
        col1 = Column(self.causal_graph, self.relationships)
        col2 = Column(self.explore_dataset, self.combinations)
        row1 = Row(col2, self.comparator)
        col3 = Column(self.overview, row1)
        ui = Row(col1, col3)
        curdoc().add_root(ui)
        return ui

class CausalGraphView(FairHIL):

    def launch_self(self):
        pass


class SystemOverview(FairHIL):
    pass


class RelationshipsView(FairHIL):
    pass


class DatasetView(FairHIL):
    pass


class CombinationView(FairHIL):
    pass


class ComparisonView(FairHIL):
    pass
