import base64
import io
from bokeh.io import *
from bokeh.models import *
from bokeh.layouts import *
import numpy as np
import pandas as pd
from configuration import *
from fairhil import *


class Wizard:

	def __init__(self):
		self.CONFIG = Configuration()  # Instantiating configuration class
		self.CONFIG.set_mode(ModeOptions.BASIC)  # Initial mode is set to basic

		# Instantiating Stage1 UI components
		self.file_input_div = Div(text="Drag and drop or click to upload CSV ML Dataset from local machine:")
		self.selection_div = Div(text="Please choose a set-up mode:")
		self.file_input = FileInput(accept=".csv")
		self.mode_button = RadioButtonGroup(labels=get_options(self.CONFIG.MODE), active=0)
		self.submit_stage_1 = Button(label="Submit", button_type="success")
		self.callback_holder = PreText(text='', css_classes=['hidden'], visible=False)

		# Defining Stage1 hooks
		self.file_input.on_change('value', self.set_dataset)
		self.mode_button.on_change('active', self.set_mode)
		self.submit_stage_1.on_click(self.launch_stage_2)
		self.callback_holder.js_on_change('text', CustomJS(args={}, code='alert(cb_obj.text);'))

		# Instantiating Stage2 UI components
		self.target_feature_div = Div(text="Select a target feature:")
		self.sensi_feats_choice_div = Div(text="Select sensitive features:")
		self.deep_dive_metrics_div = Div(text="Select a suite of deep-dive fairness metrics:")
		self.primary_metric_div = Div(text="Select a primary fairness metric:")
		self.binning_process_div = Div(text="Select a barchart binning process:")
		self.discovery_algorithm_div = Div(text="Select a causal discovery algorithm:")
		self.card_generation_div = Div(text="Select a card generation process:")

		self.target_feature = Select()
		self.sensi_feats_choice = MultiChoice()
		self.deep_dive_metrics = MultiChoice(options=get_options(FairnessMetrics), value=list(
			np.random.choice(get_options(FairnessMetrics), size=2,
							 replace=False)))  # Randomly generating 2 deep-dive metrics
		self.primary_metric = Select(options=get_options(FairnessMetrics), value=FairnessMetrics.SPD.string)
		self.binning_process = Select(options=get_options(BinningProcesses), value=BinningProcesses.SQUAREROOT.string)
		self.discovery_algorithm = Select(options=get_options(DiscoveryAlgorithms),
										  value=DiscoveryAlgorithms.DEFAULT.string)
		self.card_generation = Select(options=get_options(CardGenerationProcesses),
									  value=CardGenerationProcesses.MANUAL.string)
		self.submit_stage_2 = Button(label="Submit", button_type="success")

		# Defining Stage2 hooks
		self.submit_stage_2.on_click(self.launch_fairhil)

	def launch_stage_1(self):
		curdoc().clear()
		curdoc().title = "Wizard Stage 1"
		grid = gridplot([
			[self.file_input_div, self.file_input],
			[self.selection_div, self.mode_button],
		])
		ui = column(grid, self.submit_stage_1, self.callback_holder)
		curdoc().add_root(ui)
		return ui

	def set_dataset(self, attr, old, new):
		decoded = base64.b64decode(new)
		file = io.BytesIO(decoded)
		self.CONFIG.set_dataset(file)

	def set_mode(self, attr, old, new):
		if self.mode_button.active == 0:
			self.CONFIG.set_mode(ModeOptions.BASIC)
		elif self.mode_button.active == 1:
			self.CONFIG.set_mode(ModeOptions.ADVANCED)
		print(self.CONFIG.MODE)

	def launch_stage_2(self):
		if len(self.CONFIG.DATASET) == 0:
			print("Alert")
			self.callback_holder.text = "Please upload a Dataset"
		else:
			curdoc().clear()
			curdoc().title = "Wizard Stage 2"
			dataset_feats = list(self.CONFIG.DATASET.columns.values)
			self.CONFIG.set_dataset_feats(dataset_feats)
			self.target_feature.update(options=dataset_feats, value=dataset_feats[-1])
			self.sensi_feats_choice.update(options=dataset_feats)
			if self.CONFIG.MODE.value == 1:
				# Advanced config UI
				grid = gridplot([
					[self.sensi_feats_choice_div, self.sensi_feats_choice],
					[self.target_feature_div, self.target_feature],
					[self.deep_dive_metrics_div, self.deep_dive_metrics],
					[self.primary_metric_div, self.primary_metric],
					[self.binning_process_div, self.binning_process],
					[self.discovery_algorithm_div, self.discovery_algorithm],
					[self.card_generation_div, self.card_generation],
				])
				ui = column(grid, self.submit_stage_2)
			else:
				# Basic config UI
				grid = gridplot([
					[self.sensi_feats_choice_div, self.sensi_feats_choice],
					[self.target_feature_div, self.target_feature],
					[self.deep_dive_metrics_div, self.deep_dive_metrics],
					[self.primary_metric_div, self.primary_metric],
				])
				ui = column(grid, self.submit_stage_2)
			curdoc().add_root(ui)
			return ui

	def launch_fairhil(self):
		print("Launching FairHIL")
		# Need to set config file here!!!!!!!!!
		fh = FairHIL(self.CONFIG)
