from configuration import *
from bokeh.io import *
from bokeh.models import *
from bokeh.layouts import *
from bokeh.plotting import *


class FairHIL:

	def __init__(self, config):
		self.CONFIG = config
		self.causal_graph = CausalGraphView
		self.overview = SystemOverview
		self.relationships = RelationshipsView
		self.explore_dataset = DatasetView
		self.combinations = CombinationView
		self.comparator = ComparisonView

		print(self.CONFIG.DATASET)
		print(self.CONFIG.SENSITIVE_FEATS)

	def launch_ui(self):
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

	def launch_ui(self):
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
