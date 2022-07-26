from bokeh.io import show, curdoc
from bokeh.models.widgets import FileInput
from bokeh.models import *
from bokeh.layouts import column, row, gridplot, layout
import base64
import io
import pandas as pd

DATASET = pd.DataFrame()
FAIRNESS_METRICS = ["SPD", "Equality of Opportunity Difference", "Average odds difference", "Disparate impact", "Theill index"]
DISCOVERY_ALGORITHMS = ["Default"]
BINNING_PROCESSES = ["Square Root"]

file_input_div = Div(text="Drag and drop or click to upload CSV ML Dataset from local machine:")
file_input = FileInput(accept=".csv", width=400)
target_feature_div = Div(text="Select a target feature:")
target_feature = Select()
sensi_feats_choice_div = Div(text="Select sensitive features:")
sensi_feats_choice = MultiChoice()
deep_dive_metrics_div = Div(text="Select a suite of deep-dive fairness metrics:")
deep_dive_metrics = MultiChoice(options=FAIRNESS_METRICS)
primary_metric_div = Div(text="Select a primary fairness metric:")
primary_metric = Select(options=FAIRNESS_METRICS)
binning_process_div = Div(text="Select a barchart binning process:")
binning_process = Select(options=BINNING_PROCESSES)
discovery_algorithm_div = Div(text="Select a causal discovery algorithm:")
discovery_algorithm = Select(options=DISCOVERY_ALGORITHMS)
card_generation_div = Div(text="Select a card generation process:")
card_generation = Select(options=["Manual", "Automatic"])
submit = Button(label="Submit preferences", button_type="success")


def upload_dataset(attr, old, new):
    decoded = base64.b64decode(new)
    file = io.BytesIO(decoded)
    DATASET = pd.read_csv(file, index_col=[0])
    features = list(DATASET.columns.values)
    sensi_feats_choice.update(options=features)
    target_feature.update(options=features)


file_input.on_change('value', upload_dataset)


grid = gridplot([
    [row(file_input_div, file_input)],
    [row(sensi_feats_choice_div, sensi_feats_choice)],
    [row(target_feature_div, target_feature)],
    [row(deep_dive_metrics_div, deep_dive_metrics)],
    [row(primary_metric_div, primary_metric)],
    [row(binning_process_div, binning_process)],
    [row(discovery_algorithm_div, discovery_algorithm)],
    [row(card_generation_div, card_generation)],
    [submit]
])

curdoc().add_root(grid)
