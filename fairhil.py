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
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from aif360.sklearn.metrics import *
from aif360.datasets import *
from aif360.metrics import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from statsmodels.robust.scale import mad
import dalex as dx
from optbinning import OptimalBinning


class FairHIL:

	def __init__(self, config):
		self.CONFIG = config
		cdt.SETTINGS.rpath = "C:/Program Files/R/R-4.2.1/bin/Rscript"  # Path to Rscript.exe
		self.plot_size = 400

		self.callback_holder = PreText(text='', css_classes=['hidden'], visible=False)
		self.callback_holder.js_on_change('text', CustomJS(args={}, code='alert(cb_obj.text);'))

		# Loading/instantiating UI components
		print("Loading FairHIL interface...")
		self.ui = None
		self.overview_fig = self.load_overview_fig()
		self.causal_graph_fig = self.load_causal_graph_fig()
		self.distribution_cds, self.distribution_fig, self.distribution_data = self.load_distribution_fig()
		self.fairness_data = self.get_fairness_data()
		print(self.fairness_data)
		self.fairness_cds, self.fairness_fig = self.load_fairness_fig()
		self.relationships_fig = Spacer()
		self.dataset_cds, self.data_table_fig = self.load_dataset_fig()
		self.combinations_fig = Spacer()
		self.comparator_fig = Spacer()

		self.launch_ui()
		print("Done")

	def launch_ui(self):
		curdoc().title = "FairHIL"
		curdoc().clear()
		self.ui = layout(children=[
			[self.overview_fig],
			[self.causal_graph_fig, column(self.fairness_fig, self.distribution_fig), self.data_table_fig],
			[self.relationships_fig, self.combinations_fig, self.comparator_fig],
			[self.callback_holder]
		])
		curdoc().add_root(self.ui)

	def load_overview_fig(self):
		title_div = Div(text="<b>System Overview<b>", style={"text-align": "center", "font-size": "125%"})
		instances_div = Div(text=f"Total dataset instances: {len(self.CONFIG.DATASET)}",
							style={"text-align": "center", "font-size": "125%"})
		cats_vals = self.CONFIG.DATASET[self.CONFIG.TARGET_FEAT].value_counts()
		cats_vals_lst = []
		for cat, val in cats_vals.items():
			cats_vals_lst.append(f"{cat}: {val}")
		cats_vals_div = Div(text=f"{', '.join(cats_vals_lst)}", style={"text-align": "center", "font-size": "125%"})
		percentage_div = Div(text=f"{np.round((cats_vals.max() / cats_vals.sum()) * 100, 1)}% major class",
							 style={"text-align": "center", "font-size": "125%", "color": "blue"})
		pi_fig = self.get_pi_fig(cats_vals, 50, 50)
		return layout(children=[[title_div, instances_div, cats_vals_div, percentage_div, pi_fig]],
					  sizing_mode="stretch_both")

	def get_pi_fig(self, series, height, width):
		data = series.reset_index(name='value').rename(columns={'index': 'target'})
		data['angle'] = data['value'] / data['value'].sum() * 2 * math.pi
		data['color'] = ('#253494', '#41b6c4') if len(series) < 3 else YlGnBu[len(series)]
		pi_fig = figure(height=height, width=width, toolbar_location=None, tools="hover", tooltips="@target: @value",
						x_range=Range1d(-1, 1))
		pi_fig.wedge(x=0, y=1, radius=0.5, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
					line_color="white", fill_color='color', source=data)
		pi_fig.axis.axis_label = None
		pi_fig.axis.visible = False
		pi_fig.grid.grid_line_color = None
		pi_fig.outline_line_color = None
		return pi_fig

	def load_causal_graph_fig(self):
		if DiscoveryAlgorithms(self.CONFIG.DISCOVERY_ALG).value == 1:
			alg = cdt.causality.graph.GES()
		elif DiscoveryAlgorithms(self.CONFIG.DISCOVERY_ALG).value == 2:
			alg = cdt.causality.graph.LiNGAM()
		else:
			alg = cdt.causality.graph.PC()

		G = alg.create_graph_from_data(self.CONFIG.ENCODED_DATASET)
		plot = Plot(width=self.plot_size, height=self.plot_size, x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1),
					title="Causal Discovery Graph", title_location="left")
		plot.add_tools(HoverTool(tooltips=None), TapTool())  # PanTool(), WheelZoomTool()
		hv_graph = hv.Graph.from_networkx(G, nx.circular_layout(G, scale=0.8, center=(0, 0))).opts(directed=True)
		hv_rendered = hv.render(hv_graph, 'bokeh')
		graph_renderer = hv_rendered.select_one(GraphRenderer)

		node_normal_color = named.deepskyblue
		node_hover_color = named.violet
		node_selection_color = named.violet
		edge_normal_color = named.dimgray
		edge_hover_color = named.magenta
		edge_selection_color = named.magenta

		graph_renderer.node_renderer.data_source.data['degrees'] = [(val + 1) * 10 for (node, val) in
																	sorted(nx.degree(G), key=lambda pair: pair[0])]
		graph_renderer.node_renderer.data_source.data['colors'] = [
			named.gold if feat in self.CONFIG.SENSITIVE_FEATS else node_normal_color for feat in
			self.CONFIG.DATASET_FEATS]

		graph_renderer.node_renderer.glyph = Circle(size='degrees', fill_color='colors', fill_alpha=0.5)
		graph_renderer.node_renderer.selection_glyph = Circle(size='degrees', fill_color=node_selection_color,
															  fill_alpha=0.5)
		graph_renderer.node_renderer.hover_glyph = Circle(size='degrees', fill_color=node_hover_color, fill_alpha=0.5)

		graph_renderer.edge_renderer.glyph = MultiLine(line_color=edge_normal_color, line_width=2)
		graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=edge_selection_color, line_width=2)
		graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=edge_hover_color, line_width=2)

		graph_renderer.selection_policy = NodesAndLinkedEdges()
		graph_renderer.inspection_policy = NodesAndLinkedEdges()

		source = ColumnDataSource(
			{'x': hv_graph.nodes['x'], 'y': hv_graph.nodes['y'], 'field': self.CONFIG.ENCODED_DATASET.columns})
		labels = LabelSet(x='x', y='y', text='field', source=source, text_font_size='7pt', text_color='black',
						  text_align='center')

		graph_renderer.node_renderer.data_source.selected.on_change("indices", self.update_distribution_cds,
																	self.update_fairness_cds)

		plot.renderers.append(graph_renderer)
		plot.renderers.append(labels)

		return plot

	def load_distribution_fig(self):
		distribution_cds = ColumnDataSource(data={'top': [], 'bottom': [], 'left': [], 'right': []})

		distribution_fig = figure(width=math.floor(self.plot_size / 2), height=math.floor(self.plot_size * 0.75),
								  title="Feature distribution against target", title_location="left", tools="")
		distribution_fig.quad(top='top', bottom='bottom', left='left', right='right', source=distribution_cds,
							  line_color="white")
		distribution_fig.axis.visible = False

		distribution_data = []
		for column_idx in range(len(self.CONFIG.ENCODED_DATASET.columns)):
			hist, edges = np.histogram(self.CONFIG.ENCODED_DATASET.iloc[:, column_idx],
									   weights=self.CONFIG.ENCODED_DATASET[self.CONFIG.TARGET_FEAT],
									   bins=2 if self.CONFIG.ENCODED_DATASET.iloc[:, column_idx].nunique() == 2 else 10)
			distribution_data.append({'top': hist, 'bottom': [0] * len(hist), 'left': edges[:-1], 'right': edges[1:]})

		return distribution_cds, distribution_fig, distribution_data

	def update_distribution_cds(self, attr, old, new):
		if new:
			self.distribution_cds.data = self.distribution_data[int(new[0])]

	def load_fairness_fig(self):
		fairness_cds = ColumnDataSource(data={'y': [], 'right': []})

		fairness_fig = figure(width=math.floor(self.plot_size / 2), height=math.floor(self.plot_size * 0.25),
							  x_range=Range1d(min([d[self.CONFIG.PRIMARY_METRIC] for d in self.fairness_data.values()]),
											  max([d[self.CONFIG.PRIMARY_METRIC] for d in self.fairness_data.values()])),
							  title="Fairness", title_location="left", tools="")
		fairness_fig.hbar(y='y', right='right', left=0, height=0.5, source=fairness_cds)
		fairness_fig.yaxis.major_tick_line_color = None
		fairness_fig.yaxis.minor_tick_line_color = None
		fairness_fig.yaxis.visible = False
		fairness_fig.xaxis.major_label_orientation = math.pi / 4
		fairness_fig.yaxis.major_label_orientation = "vertical"

		return fairness_cds, fairness_fig

	def get_fairness_data(self):
		fairness_data = {}
		for column in self.CONFIG.DATASET_FEATS:
			if self.CONFIG.SENSITIVE_FEATS:
				dataset_orig = StandardDataset(self.CONFIG.ENCODED_DATASET, label_name=self.CONFIG.TARGET_FEAT,
											   favorable_classes=[self.CONFIG.TARGET_FAVOURABLE_CLASS],
											   protected_attribute_names=self.CONFIG.SENSITIVE_FEATS,
											   privileged_classes=[self.CONFIG.PRIV_CLASS_DIVIDE[feat] for feat in
																   self.CONFIG.SENSITIVE_FEATS])
				dataset_pred = dataset_orig.copy()
				y_pred = self.get_y_pred(self.CONFIG.ENCODED_DATASET[column])
				dataset_pred.labels = y_pred

				privileged_groups = [{attr: 1 for attr in self.CONFIG.SENSITIVE_FEATS}]
				unprivileged_groups = [{attr: 0 for attr in self.CONFIG.SENSITIVE_FEATS}]

				classified_metric = ClassificationMetric(dataset_orig, dataset_pred, unprivileged_groups=unprivileged_groups,
														 privileged_groups=privileged_groups)
				# binary_metric = BinaryLabelDatasetMetric(dataset_orig, unprivileged_groups=unprivileged_groups,
				# 										 privileged_groups=privileged_groups)
				result = {"Statistical Parity Difference": classified_metric.statistical_parity_difference(),
						  "Equality of Opportunity Difference": classified_metric.equal_opportunity_difference(),
						  "Average Odds Difference": classified_metric.average_abs_odds_difference(),
						  "Disparate Impact": classified_metric.disparate_impact(),
						  "Theill Index": classified_metric.theil_index()}
				fairness_data[column] = result
			else:
				fairness_data[column] = {"Statistical Parity Difference": 0,
										 "Equality of Opportunity Difference": 0,
										 "Average Odds Difference": 0,
										 "Disparate Impact": 0,
										 "Theill Index": 0}
				self.callback_holder.text = "No protected features selected: fairness metric(s) not calculated"
		return fairness_data

	def get_y_pred(self, X):
		model = xgb.XGBClassifier(objective="binary:logistic")
		y = self.CONFIG.ENCODED_DATASET[self.CONFIG.TARGET_FEAT]
		model.fit(X, y)
		return model.predict(X)

	def update_fairness_cds(self, attr, old, new):
		if new:
			self.fairness_cds.data = {'y': [1], 'right': [self.fairness_data[self.CONFIG.DATASET_FEATS[int(new[0])]][self.CONFIG.PRIMARY_METRIC]]}

	def load_dataset_fig(self):
		cols = [TableColumn(field=x, title=x) for x in self.CONFIG.DATASET.columns]
		dataset_cds = ColumnDataSource(self.CONFIG.DATASET)
		data_table = DataTable(columns=cols, source=dataset_cds, height=math.floor(0.9 * self.plot_size))
		unfair_button = Button(label="Mark data point as unfair", button_type="danger",
							   height=math.floor(0.1 * self.plot_size))

		unfair_button.on_click(self.mark_unfair)
		return dataset_cds, column(unfair_button, data_table)

	def mark_unfair(self):
		print(self.dataset_cds.selected.__getattribute__('indices'))
