from configuration import *
from bokeh.io import *
from bokeh.models import *
from bokeh.layouts import *
from bokeh.plotting import *
from bokeh.palettes import Spectral4
import cdt
import networkx as nx
import matplotlib.pyplot as plt


class FairHIL():

	def __init__(self, config):
		self.CONFIG = config
		cdt.SETTINGS.rpath = "C:/Program Files/R/R-4.2.1/bin/Rscript"
		# print(f"NORMAL:\n{self.CONFIG.DATASET.head().to_string()}\n")
		# print(f"ENCODED:\n{self.CONFIG.ENCODED_DATASET.head(75).to_string()}\n")

		# self.causal_graph = CausalGraphView
		# self.overview = SystemOverview
		# self.relationships = RelationshipsView
		# self.explore_dataset = DatasetView
		# self.combinations = CombinationView
		# self.comparator = ComparisonView

		# print(self.CONFIG.SENSITIVE_FEATS)
		# print(type(self.CONFIG.MODE))
		# print(type(self.CONFIG.DEEP_DIVE_METRICS))
		# print(type(self.CONFIG.PRIMARY_METRIC))


	def launch_ui(self):
		curdoc().title = "FairHIL"
		curdoc().clear()
		# col1 = Column(self.causal_graph, self.relationships)
		# col2 = Column(self.explore_dataset, self.combinations)
		# row1 = Row(col2, self.comparator)
		# col3 = Column(self.overview, row1)
		# ui = Row(col1, col3)

		ui = self.load_causal_graph()

		curdoc().add_root(ui)
		return ui

	def load_causal_graph(self):
		glasso = cdt.independence.graph.Glasso()
		skeleton = glasso.predict(self.CONFIG.ENCODED_DATASET, max_iter=int(1.0e+4))

		alg = cdt.causality.graph.PC()
		# if DiscoveryAlgorithms(self.CONFIG.DISCOVERY_ALG).value == 0:
		# 	print("success")
		# 	alg = cdt.causality.graph.PC()

		G = alg.predict(self.CONFIG.ENCODED_DATASET, skeleton)
		print(f"TYPE OF G IS: {type(G)}")

		# graph_nx = nx.draw_networkx(output)

		plot = Plot(width=600, height=600, x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
		plot.title.text = "Networkx Integration Demonstration"
		plot.add_tools(HoverTool(tooltips=None), TapTool(), BoxSelectTool())
		graph_renderer = from_networkx(G, nx.spring_layout, scale=1, center=(0, 0))

		# graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[0])
		# graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
		# graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])
		#
		# graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=5)
		# graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
		# graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)
		#
		# graph_renderer.selection_policy = NodesAndLinkedEdges()
		# graph_renderer.inspection_policy = EdgesAndLinkedNodes()

		plot.renderers.append(graph_renderer)
		print("done")
		return plot


class CausalGraphView(FairHIL):
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
