import aenum
import pandas as pd


class Configuration:

	def __init__(self):
		self.DATASET = pd.DataFrame()
		self.DATASET_FEATS = []
		self.MODE = ModeOptions
		self.SENSITIVE_FEATS = []
		self.TARGET_FEAT = None
		self.DEEP_DIVE_METRICS = [FairnessMetrics]
		self.PRIMARY_METRIC = FairnessMetrics
		self.BINNING_PROCESS = BinningProcesses
		self.DISCOVERY_ALG = DiscoveryAlgorithms
		self.CARD_GEN_PROCESS = CardGenerationProcesses


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
