import base64
import io
import enum
import aenum
from bokeh.io import *
from bokeh.models import *
from bokeh.layouts import *
import numpy as np
import pandas as pd


class Mode(enum.Enum):
    BASIC = 1
    ADVANCED = 2


class FairnessMetrics(aenum.Enum):
    _init_ = 'value string'
    SPD = 1, "Statistical Parity Difference"
    EoOD = 2, "Equality of Opportunity Difference"
    AOD = 3, "Average Odds Difference"
    DI = 4, "Disparate Impact"
    TI = 5, "Theill Index"

    def __str__(self):
        return self.string

    def options(self):
        return [option.string for option in self]

    @classmethod
    def _missing_value_(cls, value):
        for member in cls:
            if member.string == value:
                return member


class CausalDiscoveryAlgorithms(aenum.Enum):
    _init_ = 'value string'
    DEFAULT = 1, "Default"

    def __str__(self):
        return self.string

    @classmethod
    def _missing_value_(cls, value):
        for member in cls:
            if member.string == value:
                return member


class BinningProcesses(aenum.Enum):
    _init_ = 'value string'
    SQUAREROOT = 1, "Square Root"

    def __str__(self):
        return self.string

    @classmethod
    def _missing_value_(cls, value):
        for member in cls:
            if member.string == value:
                return member


class ModeSelection:

    def __init__(self):
        # Instantiating UI components
        self.file_input_div = Div(text="Drag and drop or click to upload CSV ML Dataset from local machine:")
        self.selection_div = Div(text="Please choose a set-up mode:")
        self.file_input = FileInput(accept=".csv")
        self.mode_button = RadioButtonGroup(labels=["BASIC", "ADVANCED"], active=0)
        self.submit = Button(label="Submit", button_type="success")
        self.callback_holder = PreText(text='', css_classes=['hidden'], visible=False)

        # Setting initial mode as BASIC and empty dataframe
        self.DATASET = pd.DataFrame()
        self.MODE = Mode.BASIC

        # Defining hooks
        self.file_input.on_change('value', self.set_dataset)
        self.mode_button.on_change('active', self.set_mode)
        self.submit.on_click(self.launch_next)
        self.callback_holder.js_on_change('text', CustomJS(args={}, code='alert(cb_obj.text);'))

    def launch_self(self):
        curdoc().title = "Wizard"
        curdoc().clear()
        grid = gridplot([
            [self.file_input_div, self.file_input],
            [self.selection_div, self.mode_button],
        ])
        ui = column(grid, self.submit, self.callback_holder)
        curdoc().add_root(ui)
        return ui

    def set_dataset(self, attr, old, new):
        decoded = base64.b64decode(new)
        file = io.BytesIO(decoded)
        self.DATASET = pd.read_csv(file, index_col=[0])

    def set_mode(self, attr, old, new):
        if self.mode_button.active == 0:
            self.MODE = Mode.BASIC
        elif self.mode_button.active == 1:
            self.MODE = Mode.ADVANCED
        print(self.MODE)

    def launch_next(self):
        if len(self.DATASET) == 0:
            print("Alert")
            self.callback_holder.text = "Please upload a Dataset"
        else:
            Configuration().launch_self(self.DATASET, self.MODE)


class Configuration:
    FAIRNESS_METRICS = FairnessMetrics
    CAUSAL_DISCOVERY_ALGORITHMS = CausalDiscoveryAlgorithms
    BINNING_PROCESSES = BinningProcesses

    DATASET = pd.DataFrame()
    MODE = Mode
    SENSI_FEATS = []
    TARGET_FEAT = None
    DEEP_DIVE_METRICS = []
    PRIMARY_METRIC = None
    BINNING = None
    CAUSAL_ALGO = None
    CARD_GEN = None

    def __init__(self):
        self.FAIRNESS_METRICS_OPTIONS = [option.string for option in self.FAIRNESS_METRICS]
        self.CAUSAL_DISCOVERY_ALGORITHMS_OPTIONS = [option.string for option in self.CAUSAL_DISCOVERY_ALGORITHMS]
        self.BINNING_PROCESSES_OPTIONS = [option.string for option in self.BINNING_PROCESSES]

        # Instantiating UI components
        self.target_feature_div = Div(text="Select a target feature:")
        self.sensi_feats_choice_div = Div(text="Select sensitive features:")
        self.deep_dive_metrics_div = Div(text="Select a suite of deep-dive fairness metrics:")
        self.primary_metric_div = Div(text="Select a primary fairness metric:")
        self.binning_process_div = Div(text="Select a barchart binning process:")
        self.discovery_algorithm_div = Div(text="Select a causal discovery algorithm:")
        self.card_generation_div = Div(text="Select a card generation process:")

        self.target_feature = Select()
        self.sensi_feats_choice = MultiChoice()
        self.deep_dive_metrics = MultiChoice(options=self.FAIRNESS_METRICS_OPTIONS,
                                             value=list(
                                                 np.random.choice(self.FAIRNESS_METRICS_OPTIONS, size=3,
                                                                  replace=False)))
        self.primary_metric = Select(options=self.FAIRNESS_METRICS_OPTIONS)
        self.binning_process = Select(options=self.BINNING_PROCESSES_OPTIONS)
        self.discovery_algorithm = Select(options=self.CAUSAL_DISCOVERY_ALGORITHMS_OPTIONS)
        self.card_generation = Select(options=["Manual", "Automatic"])
        self.submit = Button(label="Submit", button_type="success")

        # Defining hooks
        self.submit.on_click(self.launch_next)

    def launch_self(self, dataset, mode):
        curdoc().title = "Wizard"
        curdoc().clear()
        self.DATASET = dataset
        self.MODE = mode
        features = list(dataset.columns.values)
        self.target_feature.update(options=features, value=features[-1])
        self.sensi_feats_choice.update(options=features)
        if mode.value == 1:
            grid = gridplot([
                [self.sensi_feats_choice_div, self.sensi_feats_choice],
                [self.target_feature_div, self.target_feature],
                [self.deep_dive_metrics_div, self.deep_dive_metrics],
                [self.primary_metric_div, self.primary_metric],
            ])
            ui = column(grid, self.submit)
        else:
            grid = gridplot([
                [self.sensi_feats_choice_div, self.sensi_feats_choice],
                [self.target_feature_div, self.target_feature],
                [self.deep_dive_metrics_div, self.deep_dive_metrics],
                [self.primary_metric_div, self.primary_metric],
                [self.binning_process_div, self.binning_process],
                [self.discovery_algorithm_div, self.discovery_algorithm],
                [self.card_generation_div, self.card_generation],
            ])
            ui = column(grid, self.submit)
        curdoc().add_root(ui)
        return ui

    def launch_next(self):

        # Launch FairHIL here
        print("Launching FairHIL")
        return None
