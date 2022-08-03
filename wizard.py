import enum
from functools import partial
from bokeh.io import *
from bokeh.models import *
from bokeh.layouts import *
import base64
import io
import pandas as pd


class Mode(enum.Enum):
    BASIC = 1
    ADVANCED = 2


class Configure:
    FAIRNESS_METRICS = ["SPD", "Equality of Opportunity Difference", "Average odds difference", "Disparate impact",
                        "Theill index"]
    DISCOVERY_ALGORITHMS = ["Default"]
    BINNING_PROCESSES = ["Square Root"]

    target_feature_div = Div()
    sensi_feats_choice_div = Div()
    deep_dive_metrics_div = Div()
    primary_metric_div = Div()
    binning_process_div = Div()
    discovery_algorithm_div = Div()
    card_generation_div = Div()
    target_feature = Select()
    sensi_feats_choice = MultiChoice()
    deep_dive_metrics = MultiChoice()
    primary_metric = Select()
    binning_process = Select()
    discovery_algorithm = Select()
    card_generation = Select()
    submit = Button()

    def __init__(self):
        # Instantiating UI components
        self.target_feature_div = Div(text="Select a target feature:")
        self.sensi_feats_choice_div = Div(text="Select sensitive features:")
        self.deep_dive_metrics_div = Div(text="Select a suite of deep-dive fairness metrics:")
        self.primary_metric_div = Div(text="Select a primary fairness metric:")
        self.binning_process_div = Div(text="Select a barchart binning process:")
        self.discovery_algorithm_div = Div(text="Select a causal discovery algorithm:")
        self.card_generation_div = Div(text="Select a card generation process:")
        self.deep_dive_metrics = MultiChoice(options=self.FAIRNESS_METRICS)
        self.primary_metric = Select(options=self.FAIRNESS_METRICS)
        self.binning_process = Select(options=self.BINNING_PROCESSES)
        self.discovery_algorithm = Select(options=self.DISCOVERY_ALGORITHMS)
        self.card_generation = Select(options=["Manual", "Automatic"])
        self.submit = Button(label="Submit", button_type="success")

        # Defining hooks

    def launch_self(self, dataset, mode):
        features = list(dataset.columns.values)
        self.target_feature.update(options=features)
        self.sensi_feats_choice.update(options=features)
        if mode.value == 1:
            ui = gridplot([
                [row(self.sensi_feats_choice_div, self.sensi_feats_choice)],
                [row(self.target_feature_div, self.target_feature)],
                [row(self.deep_dive_metrics_div, self.deep_dive_metrics)],
                [row(self.primary_metric_div, self.primary_metric)],
                [self.submit]
            ], sizing_mode="fixed")
        else:
            ui = gridplot([
                [row(self.sensi_feats_choice_div, self.sensi_feats_choice)],
                [row(self.target_feature_div, self.target_feature)],
                [row(self.deep_dive_metrics_div, self.deep_dive_metrics)],
                [row(self.primary_metric_div, self.primary_metric)],
                [row(self.binning_process_div, self.binning_process)],
                [row(self.discovery_algorithm_div, self.discovery_algorithm)],
                [row(self.card_generation_div, self.card_generation)],
                [self.submit]
            ], sizing_mode="fixed")
        curdoc().add_root(ui)
        return ui


class ModeSelection:
    DATASET = pd.DataFrame()
    MODE = Mode

    file_input_div = Div()
    selection_div = Div()
    file_input = FileInput()
    mode_button = RadioButtonGroup()
    submit = Button()

    callback_holder = PreText(text='', css_classes=['hidden'], visible=False)

    def __init__(self):
        # Instantiating UI components
        self.file_input_div = Div(text="Drag and drop or click to upload CSV ML Dataset from local machine:")
        self.selection_div = Div(text="Please choose a set-up mode:")
        self.file_input = FileInput(accept=".csv")
        self.mode_button = RadioButtonGroup(labels=["BASIC", "ADVANCED"], active=0)
        self.submit = Button(label="Submit", button_type="success")

        # Setting initial mode as BASIC
        self.MODE = Mode.BASIC

        # Defining hooks
        self.file_input.on_change('value', self.set_dataset)
        self.mode_button.on_change('active', self.set_mode)
        self.submit.on_click(self.launch_next)

        self.callback_holder.js_on_change('text', CustomJS(args={}, code='alert(cb_obj.text);'))

    def launch_self(self):
        ui = gridplot([
            [self.file_input_div, self.file_input],
            [self.selection_div, self.mode_button],
            [self.submit],
            [self.callback_holder]
        ])
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
            curdoc().clear()
            curdoc().add_root(Configure().launch_self(self.DATASET, self.MODE))
