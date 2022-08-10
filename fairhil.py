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
from aif360.metrics import *
from aif360.datasets import *
from sklearn import metrics


class FairHIL():

	def __init__(self, config):
		self.CONFIG = config
		cdt.SETTINGS.rpath = "C:/Program Files/R/R-4.2.1/bin/Rscript"   # Path to Rscript.exe
		self.plot_size = 400

		self.ui = None

		self.get_fairness()

		# Loading/instantiating UI components
		self.overview = Spacer()
		self.causal_graph_fig = self.load_causal_graph()
		self.distribution_cds, self.distribution_fig, self.distribution_data = self.load_distribution_graph()
		self.relationships_fig = Spacer()
		self.explore_dataset_fig = Spacer()
		self.combinations_fig = Spacer()
		self.comparator_fig = Spacer()

	def launch_ui(self):
		curdoc().title = "FairHIL"
		curdoc().clear()
		self.ui = layout(children=[
			[self.overview],
			[self.causal_graph_fig, self.distribution_fig, self.explore_dataset_fig],
			[self.relationships_fig, self.combinations_fig, self.comparator_fig]
		], sizing_mode="fixed")
		curdoc().add_root(self.ui)

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

		plot = Plot(width=self.plot_size, height=self.plot_size, x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1), title="Causal Discovery Graph", title_location="left")
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

		graph_renderer.node_renderer.data_source.selected.on_change("indices", self.update_distribution_cds)

		plot.renderers.append(graph_renderer)
		plot.renderers.append(labels)

		print("done")
		return plot

	def load_distribution_graph(self):
		distribution_cds = ColumnDataSource(data={'top': [], 'bottom': [], 'left': [], 'right': []})

		distribution_fig = figure(width=math.floor(self.plot_size/2), height=math.floor(self.plot_size*0.75), title="Feature distribution against target", title_location="left", tools="")
		distribution_fig.quad(top='top', bottom='bottom', left='left', right='right', source=distribution_cds, line_color="white")
		distribution_fig.xaxis.major_label_orientation = math.pi / 4
		distribution_fig.yaxis.major_label_orientation = "vertical"

		distribution_data = []
		for column_idx in range(len(self.CONFIG.ENCODED_DATASET.columns)):
			hist, edges = np.histogram(self.CONFIG.ENCODED_DATASET.iloc[:, column_idx], weights=self.CONFIG.ENCODED_DATASET[self.CONFIG.TARGET_FEAT])
			distribution_data.append({'top': hist, 'bottom': [0]*len(hist), 'left': edges[:-1], 'right': edges[1:]})

		return distribution_cds, distribution_fig, distribution_data

	def update_distribution_cds(self, attr, old, new):
		if new:
			self.distribution_cds.data = self.distribution_data[int(new[0])]

	def get_fairness(self):
		binary_label_ds = BinaryLabelDataset(df=self.CONFIG.ENCODED_DATASETIniti, label_names=self.CONFIG.ENCODED_DATASET.columns, protected_attribute_names=self.CONFIG.SENSITIVE_FEATS)
		train, test = binary_label_ds.split(2, shuffle=True)
		metric_ds = BinaryLabelDatasetMetric(train, privileged_groups=self.CONFIG.SENSITIVE_FEATS)
		print(f"SPD: {metric_ds.mean_difference()}")

