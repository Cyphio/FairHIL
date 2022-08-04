import aenum
import pandas as pd


class Configuration:

	def __init__(self):
		self.DATASET = pd.DataFrame()
		self.DATASET_FEATS = [str]
		self.MODE = ModeOptions
		self.SENSITIVE_FEATS = [str]
		self.TARGET_FEAT = str
		self.DEEP_DIVE_METRICS = [FairnessMetrics]
		self.PRIMARY_METRIC = FairnessMetrics
		self.BINNING_PROCESS = BinningProcesses
		self.DISCOVERY_ALG = DiscoveryAlgorithms
		self.CARD_GEN_PROCESS = CardGenerationProcesses

	def set_dataset(self, file):
		self.DATASET = pd.read_csv(file, index_col=[0])

	def set_dataset_feats(self, dataset_feats):
		self.DATASET_FEATS = dataset_feats

	def set_mode(self, mode):
		self.MODE = mode

	def set_sensitive_feats(self, sensitive_feats):
		self.SENSITIVE_FEATS = sensitive_feats

	def set_target_feat(self, target_feat):
		self.TARGET_FEAT = target_feat

	def set_deep_dive_metrics(self, deep_dive_metrics):
		self.DEEP_DIVE_METRICS = deep_dive_metrics

	def set_primary_metric(self, primary_metric):
		self.PRIMARY_METRIC = primary_metric

	def set_binning_process(self, binning_process):
		self.BINNING_PROCESS = binning_process

	def set_discovery_alg(self, discovery_alg):
		self.DISCOVERY_ALG = discovery_alg

	def set_card_gen_process(self, card_gen_process):
		self.CARD_GEN_PROCESS = card_gen_process


def get_options(enum):
	return [option.string for option in enum]


def get_values(enum):
	return [option.value for option in enum]


class ModeOptions(aenum.Enum):
	_init_ = "value string"
	BASIC = 0, "Basic"
	ADVANCED = 1, "Advanced"

	def __str__(self):
		return self.string

	@classmethod
	def _missing_value_(cls, value):
		for member in cls:
			if member.string == value:
				return member


class FairnessMetrics(aenum.Enum):
	_init_ = "value string"
	SPD = 0, "Statistical Parity Difference"
	EoOD = 1, "Equality of Opportunity Difference"
	AOD = 2, "Average Odds Difference"
	DI = 3, "Disparate Impact"
	TI = 4, "Theill Index"

	def __str__(self):
		return self.string

	@classmethod
	def _missing_value_(cls, value):
		for member in cls:
			if member.string == value:
				return member


class DiscoveryAlgorithms(aenum.Enum):
	_init_ = "value string"
	DEFAULT = 0, "Default"

	def __str__(self):
		return self.string

	@classmethod
	def _missing_value_(cls, value):
		for member in cls:
			if member.string == value:
				return member


class BinningProcesses(aenum.Enum):
	_init_ = "value string"
	SQUAREROOT = 0, "Square Root"

	def __str__(self):
		return self.string

	@classmethod
	def _missing_value_(cls, value):
		for member in cls:
			if member.string == value:
				return member


class CardGenerationProcesses(aenum.Enum):
	_init_ = "value string"
	MANUAL = 0, "Manual"
	AUTOMATIC = 1, "Automatic"

	def __str__(self):
		return self.string

	@classmethod
	def _missing_value_(cls, value):
		for member in cls:
			if member.string == value:
				return member
