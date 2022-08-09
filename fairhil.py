from functools import partial

from configuration import *
from bokeh.io import *
from bokeh.models import *
from bokeh.layouts import *
from bokeh.plotting import *
from bokeh.palettes import Spectral4
from bokeh.colors import named
from bokeh import events
import cdt
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class FairHIL():

	def __init__(self, config):
		self.CONFIG = config
		cdt.SETTINGS.rpath = "C:/Program Files/R/R-4.2.1/bin/Rscript"   # Path to Rscript.exe

		self.causal_graph = self.load_causal_graph()
		self.overview = None
		self.relationships = None
		self.explore_dataset = None
		self.combinations = None
		self.comparator = None

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

		ui = self.causal_graph

		curdoc().add_root(ui)
		return ui

	def load_causal_graph(self):

		if DiscoveryAlgorithms(self.CONFIG.DISCOVERY_ALG).value == 1:
			print("GES")
			alg = cdt.causality.graph.GES()
		elif DiscoveryAlgorithms(self.CONFIG.DISCOVERY_ALG).value == 2:
			print("LiNGAM")
			alg = cdt.causality.graph.LiNGAM()
		else:
			print("PC")
			alg = cdt.causality.graph.PC()

		G = alg.create_graph_from_data(self.CONFIG.ENCODED_DATASET)

		plot = Plot(width=500, height=500, x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
		plot.title.text = "Causal Discovery Graph"
		plot.add_tools(HoverTool(tooltips=None), TapTool())    # PanTool(), WheelZoomTool()
		graph_renderer = from_networkx(G, nx.circular_layout(G, scale=0.9, center=(0, 0)), scale=1, center=(0, 0))

		node_normal_color = named.slategray
		node_hover_color = named.coral
		node_selection_color = named.olivedrab
		edge_normal_color = named.silver
		edge_hover_color = named.coral
		edge_selection_color = named.olivedrab

		graph_renderer.node_renderer.data_source.data['degrees'] = [(val+1)*10 for (node, val) in nx.degree(G)]
		graph_renderer.node_renderer.data_source.data['colors'] = [named.gold if feat in self.CONFIG.SENSITIVE_FEATS else node_normal_color for feat in self.CONFIG.DATASET_FEATS]

		graph_renderer.node_renderer.glyph = Circle(size='degrees', fill_color='colors')
		graph_renderer.node_renderer.selection_glyph = Circle(size='degrees', fill_color=node_selection_color)
		graph_renderer.node_renderer.hover_glyph = Circle(size='degrees', fill_color=node_hover_color)

		graph_renderer.edge_renderer.glyph = MultiLine(line_color=edge_normal_color, line_width=5, line_join='miter')
		graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=edge_selection_color, line_width=5, line_join='miter')
		graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=edge_hover_color, line_width=5, line_join='miter')

		graph_renderer.selection_policy = NodesAndLinkedEdges()
		graph_renderer.inspection_policy = NodesAndLinkedEdges()

		pos = graph_renderer.layout_provider.graph_layout
		x, y = zip(*pos.values())
		source = ColumnDataSource({'x': x, 'y': y, 'field': self.CONFIG.ENCODED_DATASET.columns})
		labels = LabelSet(x='x', y='y', text='field', source=source)

		graph_renderer.node_renderer.data_source.selected.on_change("indices", self.load_relationships_view)

		plot.renderers.append(graph_renderer)
		plot.renderers.append(labels)

		print("done")
		return plot

	def load_relationships_view(self, attr, old, new):
		print(self.CONFIG.ENCODED_DATASET.columns[new])


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
