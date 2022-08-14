import cdt
import networkx as nx
from bokeh.colors import named
from bokeh.layouts import *
from bokeh.models import *
from bokeh.palettes import *
from bokeh.plotting import *
from bokeh.transform import *
from configuration import *
import hvplot.networkx as hvnx
import holoviews as hv


class FairHIL:

	def __init__(self, config):
		self.CONFIG = config
		cdt.SETTINGS.rpath = "C:/Program Files/R/R-4.2.1/bin/Rscript"   # Path to Rscript.exe
		self.plot_size = 400

		self.ui = None

		# Loading/instantiating UI components
		self.overview_fig = self.load_overview_fig()
		self.causal_graph_fig = self.load_causal_graph_fig()
		self.distribution_cds, self.distribution_fig, self.distribution_data = self.load_distribution_fig()
		self.fairness_cds, self.fairness_fig, self.fairness_data = self.load_fairness_fig()
		self.relationships_fig = Spacer()
		self.dataset_cds, self.data_table_fig = self.load_dataset_fig()
		self.combinations_fig = Spacer()
		self.comparator_fig = Spacer()

		self.launch_ui()

	def launch_ui(self):
		curdoc().title = "FairHIL"
		curdoc().clear()
		self.ui = layout(children=[
			[self.overview_fig],
			[self.causal_graph_fig, column(self.fairness_fig, self.distribution_fig), self.data_table_fig],
			[self.relationships_fig, self.combinations_fig, self.comparator_fig]
		])
		curdoc().add_root(self.ui)

	def load_overview_fig(self):
		# title_div = Div(text='<b>System Overview<b>', style={'font-size': '150%'})
		title_div = Div(text="<b>System Overview<b>", style={"text-align": "center", 'font-size': '125%'})
		instances_div = Div(text=f"<b>Total dataset instances: {len(self.CONFIG.DATASET)}<b>", style={"text-align": "center", 'font-size': '125%'})
		cats_vals = self.CONFIG.DATASET[self.CONFIG.TARGET_FEAT].value_counts()
		cats_vals_lst = []
		for cat, val in cats_vals.items():
			cats_vals_lst.append(f"{cat}: {val}")
		cats_vals_div = Div(text=f"<b>{', '.join(cats_vals_lst)}<b>", style={"text-align": "center", 'font-size': '125%'})
		pi_fig = self.get_pi_fig(cats_vals, 50, 50)
		return layout(children=[[title_div, Spacer(), instances_div, Spacer(), cats_vals_div, Spacer(), pi_fig]], sizing_mode="stretch_both")

	def get_pi_fig(self, series, height, width):
		data = series.reset_index(name='value').rename(columns={'index': 'target'})
		data['angle'] = data['value']/data['value'].sum() * 2*math.pi
		data['color'] = ('#253494', '#41b6c4') if len(series) < 3 else YlGnBu[len(series)]
		pi_fig = figure(height=height, width=width, toolbar_location=None, tools="hover", tooltips="@target: @value", x_range=Range1d(-1, 1))
		pi_fig.wedge(x=0, y=1, radius=0.5, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'), line_color="white", fill_color='color', source=data)
		pi_fig.axis.axis_label = None
		pi_fig.axis.visible = False
		pi_fig.grid.grid_line_color = None
		pi_fig.outline_line_color = None
		return pi_fig

	def load_causal_graph_fig(self):
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
		hv_graph = hv.Graph.from_networkx(G, nx.circular_layout(G, scale=0.8, center=(0, 0))).opts(directed=True)
		hv_rendered = hv.render(hv_graph, 'bokeh')
		graph_renderer = hv_rendered.select_one(GraphRenderer)

		node_normal_color = named.deepskyblue
		node_hover_color = named.violet
		node_selection_color = named.violet
		edge_normal_color = named.dimgray
		edge_hover_color = named.magenta
		edge_selection_color = named.magenta

		graph_renderer.node_renderer.data_source.data['degrees'] = [(val+1)*10 for (node, val) in sorted(nx.degree(G), key=lambda pair: pair[0])]
		graph_renderer.node_renderer.data_source.data['colors'] = [named.gold if feat in self.CONFIG.SENSITIVE_FEATS else node_normal_color for feat in self.CONFIG.DATASET_FEATS]

		graph_renderer.node_renderer.glyph = Circle(size='degrees', fill_color='colors', fill_alpha=0.5)
		graph_renderer.node_renderer.selection_glyph = Circle(size='degrees', fill_color=node_selection_color, fill_alpha=0.5)
		graph_renderer.node_renderer.hover_glyph = Circle(size='degrees', fill_color=node_hover_color, fill_alpha=0.5)

		graph_renderer.edge_renderer.glyph = MultiLine(line_color=edge_normal_color, line_width=2)
		graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=edge_selection_color, line_width=2)
		graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=edge_hover_color, line_width=2)

		graph_renderer.selection_policy = NodesAndLinkedEdges()
		graph_renderer.inspection_policy = NodesAndLinkedEdges()

		source = ColumnDataSource({'x': hv_graph.nodes['x'], 'y': hv_graph.nodes['y'], 'field': self.CONFIG.ENCODED_DATASET.columns})
		labels = LabelSet(x='x', y='y', text='field', source=source, text_font_size='7pt', text_color='black', text_align='center')

		graph_renderer.node_renderer.data_source.selected.on_change("indices", self.update_distribution_cds, self.update_fairness_cds)

		plot.renderers.append(graph_renderer)
		plot.renderers.append(labels)

		print("done")
		return plot

	def load_distribution_fig(self):
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

	def load_fairness_fig(self):
		fairness_cds = ColumnDataSource(data={'y': [], 'right': []})

		fairness_fig = figure(width=math.floor(self.plot_size/2), height=math.floor(self.plot_size*0.25), x_range=Range1d(0, 1), title="Fairness", title_location="left", tools="")
		fairness_fig.hbar(y='y', right='right', left=0, height=0.5, source=fairness_cds)
		fairness_fig.yaxis.major_tick_line_color = None
		fairness_fig.yaxis.minor_tick_line_color = None

		fairness_data = []
		for column_idx in range(len(self.CONFIG.ENCODED_DATASET.columns)):
			val = self.get_fairness_value(column_idx)
			fairness_data.append({'y': [1], 'right': [val]})

		return fairness_cds, fairness_fig, fairness_data

	def get_fairness_value(self, column_idx):
		# binary_label_ds = BinaryLabelDataset(df=self.CONFIG.ENCODED_DATASET, label_names=self.CONFIG.ENCODED_DATASET.columns, protected_attribute_names=self.CONFIG.SENSITIVE_FEATS)
		# binary_label_ds = BinaryLabelDataset(df=self.CONFIG.ENCODED_DATASET, label_names=['label'],
		#                                      protected_attribute_names=self.CONFIG.SENSITIVE_FEATS)
		# train, test = binary_label_ds.split(2, shuffle=True)
		# metric_ds = BinaryLabelDatasetMetric(train, privileged_groups=self.CONFIG.SENSITIVE_FEATS)

		# standard_ds = StandardDataset(self.CONFIG.ENCODED_DATASET, label_name=self.CONFIG.TARGET_FEAT, favorable_classes=[1], protected_attribute_names=self.CONFIG.SENSITIVE_FEATS, privileged_classes=[[1]])
		# metric = BinaryLabelDatasetMetric(df=standard_ds, label_names=self.CONFIG.ENCODED_DATASET.columns, protected_attribute_names=self.CONFIG.SENSITIVE_FEATS)
		# print(f"SPD: {metric.mean_difference()}")
		return column_idx*0.1

	def update_fairness_cds(self, attr, old, new):
		if new:
			print("Update to fairness_cds")
			self.fairness_cds.data = self.fairness_data[int(new[0])]

	def load_dataset_fig(self):
		cols = [TableColumn(field=x, title=x) for x in self.CONFIG.DATASET.columns]
		dataset_cds = ColumnDataSource(self.CONFIG.DATASET)
		data_table = DataTable(columns=cols, source=dataset_cds, height=math.floor(0.9*self.plot_size))
		unfair_button = Button(label="Mark data point as unfair", button_type="danger", height=math.floor(0.1*self.plot_size))

		unfair_button.on_click(self.mark_unfair)
		return dataset_cds, column(unfair_button, data_table)

	def mark_unfair(self):

		print(self.dataset_cds.selected.__getattribute__('indices'))

