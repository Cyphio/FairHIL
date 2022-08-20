import itertools

import cdt
import networkx as nx
from bokeh.colors import named
from bokeh.layouts import *
from bokeh.models import *
from bokeh.palettes import *
from bokeh.plotting import *
from bokeh.transform import *
from configuration import *
import holoviews as hv
import xgboost as xgb
from aif360.datasets import *
from aif360.metrics import *
hv.extension("bokeh")

class FairHIL:

	def __init__(self, config):
		self.CONFIG = config
		cdt.SETTINGS.rpath = "C:/Program Files/R/R-4.2.1/bin/Rscript"  # Path to Rscript.exe
		self.plot_size = 400

		# Loading/instantiating UI components
		print("Loading FairHIL interface...")
		self.overview_fig = self.load_overview_fig()
		self.causal_graph_fig = self.load_causal_graph_fig()
		self.distribution_data = self.get_distribution_data()
		self.distribution_cds, self.distribution_fig  = self.load_distribution_fig()
		self.fairness_data = self.get_fairness_data()
		self.primary_fairness_cds, self.primary_fairness_fig = self.load_primary_fairness_fig()
		self.fairness_metrics_cds, self.relationships_fig = self.load_relationships_fig()
		self.dataset_cds, self.data_table_fig = self.load_dataset_fig()
		self.combinations_fig = Spacer()
		self.comparator_fig = Spacer()

		self.launch_ui()
		print("Done")

	def launch_ui(self):
		curdoc().title = "FairHIL"
		curdoc().clear()
		# ui = layout(children=[
		# 	[self.overview_fig, self.data_table_fig],
		# 	[self.causal_graph_fig, column(self.primary_fairness_fig, self.distribution_fig), self.relationships_fig],
		# ])
		ui = layout(children=[
			[self.overview_fig],
			[self.causal_graph_fig, column(self.primary_fairness_fig, self.distribution_fig), self.data_table_fig],
			[self.relationships_fig]
		])
		curdoc().add_root(ui)

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
		if self.CONFIG.DISCOVERY_ALG.value == 1:
			alg = cdt.causality.graph.GES()
		elif self.CONFIG.DISCOVERY_ALG.value == 2:
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

		graph_renderer.node_renderer.data_source.data['size'] = [(val + 1) * 10 for (node, val) in sorted(nx.degree(G), key=lambda pair: pair[0])]
		graph_renderer.node_renderer.data_source.data['fill_color'] = [named.gold if feat in self.CONFIG.SENSITIVE_FEATS else named.green if feat == self.CONFIG.TARGET_FEAT else node_normal_color for feat in self.CONFIG.DATASET_FEATS]

		graph_renderer.node_renderer.glyph = Circle(size='size', fill_color='fill_color', fill_alpha=0.5)
		graph_renderer.node_renderer.selection_glyph = Circle(size='size', fill_color=node_selection_color, fill_alpha=0.5)
		graph_renderer.node_renderer.hover_glyph = Circle(size='size', fill_color=node_hover_color, fill_alpha=0.5)

		graph_renderer.edge_renderer.glyph = MultiLine(line_color=edge_normal_color, line_width=2)
		graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=edge_selection_color, line_width=2)
		graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=edge_hover_color, line_width=2)

		graph_renderer.selection_policy = NodesAndLinkedEdges()
		source = ColumnDataSource(
			{'x': hv_graph.nodes['x'], 'y': hv_graph.nodes['y'], 'field': self.CONFIG.ENCODED_DATASET.columns})
		labels = LabelSet(x='x', y='y', text='field', source=source, text_font_size='7pt', text_color='black',
						  text_align='center')

		graph_renderer.node_renderer.data_source.selected.on_change("indices", self.update_distribution_cds,
																	self.update_fairness_cds,
																	self.update_fairness_metrics_cds)

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

		# for column_idx in range(len(self.CONFIG.ENCODED_DATASET.columns)):
		# 	hist, edges = np.histogram(self.CONFIG.ENCODED_DATASET.iloc[:, column_idx],
		# 							   weights=self.CONFIG.ENCODED_DATASET[self.CONFIG.TARGET_FEAT],
		# 							   bins=2 if self.CONFIG.ENCODED_DATASET.iloc[:, column_idx].nunique() == 2 else 10)
		# 	distribution_data.append({'top': hist, 'bottom': [0] * len(hist), 'left': edges[:-1], 'right': edges[1:]})

		return distribution_cds, distribution_fig

	def get_distribution_data(self):
		distribution_data = {}
		for feat in self.CONFIG.DATASET_FEATS:
			# self.CONFIG.ENCODED_DATASET['bin'] = pd.cut(self.CONFIG.ENCODED_DATASET[feat], bins=2 if feat in self.CONFIG.BINARY_FEATS else 8).astype(str)
			# df = self.CONFIG.ENCODED_DATASET.groupby('bin').bin.count().rename_axis('count').reset_index()
			# df.columns = ['bin', 'count']
			# distribution_data[feat] = df
			distribution, counts = pd.cut(self.CONFIG.ENCODED_DATASET[feat], bins=2 if feat in self.CONFIG.BINARY_FEATS else 8, retbins=True)
			print(f"DISTRIBUTION: {distribution}\nCOUNTS: {counts}")
			distribution_data[feat] = [distribution, counts]
		return distribution_data

	def update_distribution_cds(self, attr, old, new):
		if new:
			# df = self.distribution_data[self.CONFIG.DATASET_FEATS[int(new[0])]]
			# hist, edges = df['count'].values, df['bin'].values
			# print(f"X: {edges[:-1]}, Y: {edges[1:]}")

			data = self.distribution_data[self.CONFIG.DATASET_FEATS[int(new[0])]]
			self.distribution_cds.data = {'top': data[0], 'bottom': [0] * len(data[0]), 'left': data[1][:-1], 'right': data[1][1:]}
			# self.distribution_cds.data = self.distribution_data[int(new[0])]

	def load_primary_fairness_fig(self):
		primary_fairness_cds = ColumnDataSource(data={'y': [], 'right': []})

		primary_fairness_fig = figure(width=math.floor(self.plot_size / 2), height=math.floor(self.plot_size * 0.25),
							  x_range=Range1d(min([d[self.CONFIG.PRIMARY_METRIC.string] for d in self.fairness_data.values()]),
											  max([d[self.CONFIG.PRIMARY_METRIC.string] for d in self.fairness_data.values()])),
							  title=self.CONFIG.PRIMARY_METRIC.acronym, title_location="left", tools="")
		primary_fairness_fig.hbar(y='y', right='right', left=0, height=0.5, source=primary_fairness_cds)
		primary_fairness_fig.yaxis.major_tick_line_color = None
		primary_fairness_fig.yaxis.minor_tick_line_color = None
		primary_fairness_fig.yaxis.visible = False
		primary_fairness_fig.xaxis.major_label_orientation = math.pi / 4
		primary_fairness_fig.yaxis.major_label_orientation = "vertical"

		return primary_fairness_cds, primary_fairness_fig

	def get_fairness_data(self):
		fairness_data = {}
		for column in self.CONFIG.DATASET_FEATS:
			if self.CONFIG.SENSITIVE_FEATS:
				dataset_orig = StandardDataset(self.CONFIG.ENCODED_DATASET, label_name=self.CONFIG.TARGET_FEAT,
											   favorable_classes=[self.CONFIG.TARGET_FAVOURABLE_CLASS],
											   protected_attribute_names=self.CONFIG.SENSITIVE_FEATS,
											   privileged_classes=[self.CONFIG.PRIVILEGED_CLASSES[feat] for feat in
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
				# Removing NaN values from metrics (a particular issue with Disparate Impact
				cleaned_result = {k: 0 if np.isnan(v) else v for k, v in result.items()}
				fairness_data[column] = cleaned_result
			else:
				fairness_data[column] = {"Statistical Parity Difference": 0,
										 "Equality of Opportunity Difference": 0,
										 "Average Odds Difference": 0,
										 "Disparate Impact": 0,
										 "Theill Index": 0}
		return fairness_data

	def get_y_pred(self, X):
		model = xgb.XGBClassifier(objective="binary:logistic")
		y = self.CONFIG.ENCODED_DATASET[self.CONFIG.TARGET_FEAT]
		model.fit(X, y)
		return model.predict(X)

	def update_fairness_cds(self, attr, old, new):
		if new:
			self.primary_fairness_cds.data = {'y': [1], 'right': [self.fairness_data[self.CONFIG.DATASET_FEATS[int(new[0])]][self.CONFIG.PRIMARY_METRIC.string]]}

	def load_relationships_fig(self):
		fairness_metrics_cds = ColumnDataSource(data={'x': [], 'top': []})

		# x_range=list(list(self.fairness_data.values())[0].keys())
		fairness_metrics_fig = figure(width=self.plot_size, height=math.floor(self.plot_size / 2),
									  x_range=[metric.acronym for metric in self.CONFIG.DEEP_DIVE_METRICS],
									  # y_range=Range1d(
										#   min([min(list(d.values())) for d in [self.fairness_data[metric] for metric in self.CONFIG.DEEP_DIVE_METRICS]]),
										#   max([max(list(d.values())) for d in [self.fairness_data[metric] for metric in self.CONFIG.DEEP_DIVE_METRICS]])),
									  y_range=Range1d(
										  min([d[metric.string] for d in self.fairness_data.values() for metric in self.CONFIG.DEEP_DIVE_METRICS]),
										  max([d[metric.string] for d in self.fairness_data.values() for metric in self.CONFIG.DEEP_DIVE_METRICS])
									  ),
									  title="Fairness metrics deep-dive", title_location="above", tools="")
		fairness_metrics_fig.vbar(x='x', top='top', bottom=0, width=0.5, source=fairness_metrics_cds)

		relationships_fig = layout(children=[
			[fairness_metrics_fig]
		])

		return fairness_metrics_cds, relationships_fig

	def update_fairness_metrics_cds(self, attr, old, new):
		if new:
			data = self.fairness_data[self.CONFIG.DATASET_FEATS[int(new[0])]]
			self.fairness_metrics_cds.data = {'x': [metric.acronym for metric in self.CONFIG.DEEP_DIVE_METRICS], 'top': [data[metric.string] for metric in self.CONFIG.DEEP_DIVE_METRICS]}

	def load_dataset_fig(self):
		cols = [TableColumn(field=x, title=x) for x in self.CONFIG.DATASET.columns]
		dataset_cds = ColumnDataSource(self.CONFIG.DATASET)
		data_table = DataTable(columns=cols, source=dataset_cds, height=math.floor(0.9 * self.plot_size))
		unfair_button = Button(label="Mark data point as unfair", button_type="danger",
							   height=math.floor(0.1 * self.plot_size))

		unfair_button.on_click(self.mark_unfair)
		return dataset_cds, column(unfair_button, data_table)

	def mark_unfair(self):
		print(f"ROW: {self.dataset_cds.selected.__getattribute__('indices')} MARKED UNFAIR")
