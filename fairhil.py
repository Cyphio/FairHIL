from functools import partial

from bokeh.transform import linear_cmap

from configuration import *
from bokeh.io import *
from bokeh.models import *
from bokeh.layouts import *
from bokeh.plotting import *
from bokeh.palettes import *
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

		self.ui = None
		self.overview = Spacer()
		self.causal_graph = self.load_causal_graph()
		self.distribution = Spacer()
		self.relationships = Spacer()
		self.explore_dataset = Spacer()
		self.combinations = Spacer()
		self.comparator = Spacer()

	def launch_ui(self):
		curdoc().title = "FairHIL"
		curdoc().clear()
		# col1 = Column(self.causal_graph, self.relationships)
		# col2 = Column(self.distribution, self.explore_dataset, self.combinations)
		# row1 = Row(col2, self.comparator)
		# col3 = Column(self.overview, row1)
		# self.ui = Row(col1, col3)

		# self.ui = layout([
		# 	[self.overview],
		# 	[self.causal_graph, self.distribution, self.explore_dataset],
		# 	[self.relationships, self.combinations, self.comparator]
		# ])

		self.ui = layout(children=[
			[self.overview],
			[self.causal_graph, self.distribution, self.explore_dataset],
			[self.relationships, self.combinations, self.comparator]
		], sizing_mode="fixed")
		print(self.ui.children)

		# self.ui = Column(self.causal_graph, self.relationships)
		curdoc().add_root(self.ui)
		return self.ui

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

		plot = Plot(width=400, height=400, x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
		plot.title.text = "Causal Discovery Graph"
		plot.add_tools(HoverTool(tooltips=None), TapTool())    # PanTool(), WheelZoomTool()
		graph_renderer = from_networkx(G, nx.circular_layout(G, scale=0.9, center=(0, 0)), scale=1, center=(0, 0))

		node_normal_color = named.silver
		node_hover_color = named.tomato
		node_selection_color = named.tomato
		edge_normal_color = named.silver
		edge_hover_color = named.coral
		edge_selection_color = named.coral

		graph_renderer.node_renderer.data_source.data['degrees'] = [(val+1)*10 for (node, val) in nx.degree(G)]
		graph_renderer.node_renderer.data_source.data['colors'] = [named.gold if feat in self.CONFIG.SENSITIVE_FEATS else node_normal_color for feat in self.CONFIG.DATASET_FEATS]

		graph_renderer.node_renderer.glyph = Circle(size='degrees', fill_color='colors', )
		graph_renderer.node_renderer.selection_glyph = Circle(size='degrees', fill_color=node_selection_color)
		graph_renderer.node_renderer.hover_glyph = Circle(size='degrees', fill_color=node_hover_color)

		graph_renderer.edge_renderer.glyph = MultiLine(line_color=edge_normal_color, line_width=5)
		graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=edge_selection_color, line_width=5)
		graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=edge_hover_color, line_width=5)

		graph_renderer.selection_policy = NodesAndLinkedEdges()
		graph_renderer.inspection_policy = NodesAndLinkedEdges()

		pos = graph_renderer.layout_provider.graph_layout
		x, y = zip(*pos.values())
		source = ColumnDataSource({'x': x, 'y': y, 'field': self.CONFIG.ENCODED_DATASET.columns})
		labels = LabelSet(x='x', y='y', text='field', source=source)

		graph_renderer.node_renderer.data_source.selected.on_change("indices", self.update_relationships_view)

		plot.renderers.append(graph_renderer)
		plot.renderers.append(labels)

		print("done")
		return plot


	def update_relationships_view(self, attr, old, new):
		if new[0] is not None:
			hist, edges = np.histogram(self.CONFIG.ENCODED_DATASET.iloc[:, int(new[0])], weights=self.CONFIG.ENCODED_DATASET[self.CONFIG.TARGET_FEAT])
			f = figure(width=200, height=200)
			f.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white")
			self.ui.children[1].children[1] = f

