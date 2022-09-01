import base64
import io
import itertools
from fairhil import *
import re


class Wizard:

	def __init__(self):
		self.CONFIG = Configuration()  # Instantiating configuration class
		self.CONFIG.MODE = ModeOptions.BASIC  # Initial mode is set to basic

		# Instantiating Stage1 UI components
		self.file_input_div = Div(text="Drag and drop or click to upload any <b>binary-labelled CSV</b> ML Dataset from local machine:")
		self.selection_div = Div(text="Choose a set-up mode:")
		self.file_input = FileInput(accept=".csv")
		self.mode_button = RadioButtonGroup(labels=get_options(ModeOptions), active=0)
		self.submit_stage_1 = Button(label="Go to next stage of set-up", button_type="success")
		self.alert_callback_holder = PreText(text='', css_classes=['hidden'], visible=False)

		# Defining Stage1 hooks
		self.file_input.on_change('value', self.set_dataset)
		self.mode_button.on_change('active', self.set_mode)
		self.submit_stage_1.on_click(self.launch_stage_2)
		self.alert_callback_holder.js_on_change('text', CustomJS(args={}, code='alert(cb_obj.text);'))

		# Instantiating Stage2 UI components
		self.target_feature_div = Div(text="Select a target feature:")
		self.sensi_feats_choice_div = Div(text="Select the sensitive (protected) features:")
		self.priv_classes_div = Div(text="Define the privileged group boundary(s):")
		self.priv_classes_format = Div(text="<b>Format:</b> <i>feature</i> ( < / <= / > / >= / == ) <i>value<i/>")
		self.priv_classes_eg = Div(text="<b>For example:</b> Age >= 25, Sex == male")
		self.deep_dive_metrics_div = Div(text="Select a suite of deep-dive fairness metrics:")
		self.primary_metric_div = Div(text="Select a primary fairness metric:")
		self.binning_process_div = Div(text="Select a barchart binning process:")
		self.discovery_algorithm_div = Div(text="Select a causal discovery algorithm:")
		self.card_generation_div = Div(text="Select a card generation process:")

		self.target_feature = Select()
		self.sensi_feats_choice = MultiChoice()
		self.priv_classes_input = TextAreaInput(height=85)
		self.primary_metric = Select(options=get_options(FairnessMetrics), value=FairnessMetrics.SPD.string)
		self.deep_dive_metrics = MultiChoice(options=get_options(FairnessMetrics), value=get_options(FairnessMetrics))
		self.binning_process = Select(options=get_options(BinningProcesses), value=BinningProcesses.SQUAREROOT.string)
		self.discovery_algorithm = Select(options=get_options(DiscoveryAlgorithms),
										  value=DiscoveryAlgorithms.PC.string)
		self.card_generation = Select(options=get_options(CardGenerationProcesses),
									  value=CardGenerationProcesses.MANUAL.string)
		self.submit_stage_2 = Button(label="Load FairHIL", button_type="success")

		# Defining Stage2 hooks
		self.submit_stage_2.on_click(self.launch_fairhil)

	def set_dataset(self, attr, old, new):
		# Loading dataset into Pandas Dataframe
		decoded = base64.b64decode(new)
		file = io.BytesIO(decoded)
		self.CONFIG.DATASET = pd.read_csv(file, index_col=[0])

		# Hardcoded dictionary of special characters to be removed from uploaded dataset so to sanitise it and prevent interference with regex
		special_characters = {'*': 'Star', '+': 'Plus', '?': 'Q-mark', '/': 'Forward-slash'}
		self.CONFIG.DATASET.replace(special_characters, regex=False, inplace=True)

		# Encoding dataset (i.e. metricising categorical columns)
		self.CONFIG.ENCODED_DATASET = self.CONFIG.DATASET.copy()
		cat_columns = self.CONFIG.ENCODED_DATASET.select_dtypes(['object']).columns
		encoding_mapping = {}
		for col in cat_columns:
			labels, uniques = pd.factorize(self.CONFIG.ENCODED_DATASET[col])
			encoding_mapping[col] = {unique: label for label, unique in zip(set(labels), set(uniques))}
			self.CONFIG.ENCODED_DATASET[col] = labels
		self.CONFIG.ENCODING_MAPPING = encoding_mapping

		cols = [TableColumn(field=x, title=x) for x in self.CONFIG.DATASET.columns]
		dataset_cds = ColumnDataSource(self.CONFIG.DATASET)
		self.data_table = DataTable(columns=cols, source=dataset_cds)

	def set_mode(self, attr, old, new):
		if self.mode_button.active == 0:
			self.CONFIG.MODE = ModeOptions.BASIC
		elif self.mode_button.active == 1:
			self.CONFIG.MODE = ModeOptions.ADVANCED

	def get_privileged_classes(self):
		pattern = re.compile(f"({'|'.join([feat for feat in self.CONFIG.DATASET_FEATS])}{1}) (<|<=|>|>=|==) ([0-9]+|{'|'.join(list(itertools.chain.from_iterable([np.unique(self.CONFIG.DATASET[feat].values.astype(str)) for feat in self.CONFIG.DATASET_FEATS])))}{1})")
		return re.findall(pattern, str(self.priv_classes_input.value))

	def privileged_classes_complete(self, privileged_classes):
		if privileged_classes is None:
			return False
		else:
			return all(feat in [i[0] for i in privileged_classes] for feat in self.sensi_feats_choice.value)

	def privileged_classes_to_lambda(self, privileged_classes):
		return {pc[0]: eval(f"lambda x: x {' '.join([pc[1], pc[2]])}") if pc[2].isdigit() else
				eval(f"lambda x: x {' '.join([pc[1], str(self.CONFIG.ENCODING_MAPPING[pc[0]][pc[2]])])}")
				for pc in privileged_classes}

	def launch_stage_1(self):
		curdoc().clear()
		curdoc().title = "Wizard Stage 1"
		ui = layout(children=[
			[column(self.file_input_div, self.selection_div), column(self.file_input, self.mode_button)],
			[self.submit_stage_1],
			[self.alert_callback_holder]
		], sizing_mode='stretch_width')
		curdoc().add_root(ui)

	def launch_stage_2(self):
		if len(self.CONFIG.DATASET) == 0:
			self.alert_callback_holder.text = "Please upload a Dataset"
		else:
			curdoc().clear()
			curdoc().title = "Wizard Stage 2"
			dataset_feats = list(self.CONFIG.DATASET.columns.values)
			self.CONFIG.DATASET_FEATS = dataset_feats
			binary_feats = [feat for feat in dataset_feats if self.CONFIG.DATASET[feat].nunique() == 2]
			self.CONFIG.BINARY_FEATS = binary_feats
			self.target_feature.update(options=binary_feats, value=dataset_feats[-1])
			self.sensi_feats_choice.update(options=dataset_feats)
			if self.CONFIG.MODE.value == 1:
				# Advanced config UI
				ui = layout(children=[
					[self.data_table],
					[self.sensi_feats_choice_div, self.sensi_feats_choice],
					[column(self.priv_classes_div, self.priv_classes_format, self.priv_classes_eg), self.priv_classes_input],
					[self.target_feature_div, self.target_feature],
					[self.primary_metric_div, self.primary_metric],
					[self.deep_dive_metrics_div, self.deep_dive_metrics],
					# [self.binning_process_div, self.binning_process],
					[self.discovery_algorithm_div, self.discovery_algorithm],
					# [self.card_generation_div, self.card_generation],
					[self.submit_stage_2],
					[self.alert_callback_holder]
				], sizing_mode='stretch_width')
			else:
				# Basic config UI
				ui = layout(children=[
					[self.data_table],
					[self.sensi_feats_choice_div, self.sensi_feats_choice],
					[column(self.priv_classes_div, self.priv_classes_format, self.priv_classes_eg), self.priv_classes_input],
					[self.target_feature_div, self.target_feature],
					[self.primary_metric_div, self.primary_metric],
					[self.deep_dive_metrics_div, self.deep_dive_metrics],
					[self.submit_stage_2],
					[self.alert_callback_holder]
				], sizing_mode='stretch_width')
			curdoc().add_root(ui)

	def launch_fairhil(self):
		privileged_classes = self.get_privileged_classes()
		if len(self.sensi_feats_choice.value) == 0:
			self.alert_callback_holder.text = "No protected features selected" \
										"\n\nMust select at least one protected feature and define the privileged group"
		elif len(self.deep_dive_metrics.value) == 0:
			self.alert_callback_holder.text = "No deep-dive metrics selected" \
										"\n\nMust select at least one deep-dive metric"
		elif len(self.sensi_feats_choice.value) > 0 and not self.privileged_classes_complete(privileged_classes):
			self.alert_callback_holder.text = "Defined privileged group is not in correct format OR there are missing group boundary definitions" \
										"\n\nCan't launch FairHIL"
		else:
			self.CONFIG.PRIVILEGED_CLASSES = self.privileged_classes_to_lambda(privileged_classes)
			self.CONFIG.SENSITIVE_FEATS = self.sensi_feats_choice.value
			self.CONFIG.TARGET_FEAT = self.target_feature.value
			self.CONFIG.PRIMARY_METRIC = FairnessMetrics(self.primary_metric.value)
			self.CONFIG.DEEP_DIVE_METRICS = [FairnessMetrics(value) for value in self.deep_dive_metrics.value]
			self.CONFIG.BINNING_PROCESS = BinningProcesses(self.binning_process.value)
			self.CONFIG.DISCOVERY_ALG = DiscoveryAlgorithms(self.discovery_algorithm.value)
			self.CONFIG.CARD_GEN_PROCESS = CardGenerationProcesses(self.card_generation.value)
			FairHIL(self.CONFIG)
