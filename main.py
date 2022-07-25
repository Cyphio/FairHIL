from bokeh.io import show, curdoc
from bokeh.models.widgets import FileInput
from bokeh.models import ColumnDataSource, DataTable, TableColumn, CustomJS, MultiChoice
from bokeh.layouts import column, row, layout
import base64
import io
import pandas as pd

DATASET = pd.DataFrame()
SENSI_FEATS = []
FAIRNESS_METRICS = ["SPD"]

file_input = FileInput(accept=".csv", width=400)
sensi_feats_choice = MultiChoice(options=SENSI_FEATS)


def upload_dataset(attr, old, new):
    decoded = base64.b64decode(new)
    file = io.BytesIO(decoded)
    DATASET = pd.read_csv(file)
    SENSI_FEATS = DATASET.columns.values
    sensi_feats_choice.update()
    print(SENSI_FEATS)
    print("Uploaded dataset successfully")


file_input.on_change('value', upload_dataset)
sensi_feats_choice.js_on_change("value", CustomJS(code="""
    console.log('multi_choice: value=' + this.value, this.toString())
"""))


curdoc().add_root(column(file_input, sensi_feats_choice))
