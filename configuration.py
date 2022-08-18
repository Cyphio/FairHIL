import aenum
import pandas as pd


class Configuration:

	def __init__(self):
		self.DATASET = pd.DataFrame()
		self.ENCODED_DATASET = pd.DataFrame()
		self.DATASET_FEATS = []
		self.BINARY_FEATS = []
		self.MODE = ModeOptions
		self.SENSITIVE_FEATS = []
		self.TARGET_FEAT = None
		self.DEEP_DIVE_METRICS = [FairnessMetrics]
		self.PRIMARY_METRIC = FairnessMetrics
		self.BINNING_PROCESS = BinningProcesses
		self.DISCOVERY_ALG = DiscoveryAlgorithms
		self.CARD_GEN_PROCESS = CardGenerationProcesses
		self.PRIV_CLASS_DIVIDE = {"Age": lambda x: x >= 25, "Sex": [1]}
		self.TARGET_FAVOURABLE_CLASS = 1


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
	PC = 0, "PC: Peter Spirtes & Clark Glymour"
	GES = 1, "GES: Greedy Equivalence Search"
	LINGAM = 2, "LiNGAM: Linear Non-Gaussian Acyclic Model"

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
