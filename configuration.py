import aenum
import pandas as pd


class Configuration:

	def __init__(self):
		self.DATASET = pd.DataFrame()
		self.ENCODED_DATASET = pd.DataFrame()
		self.ENCODING_MAPPING = {}
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
		self.PRIVILEGED_CLASSES = {}
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
	_init_ = "value string acronym"
	SPD = 0, "Statistical Parity Difference", "SPD"
	EoOD = 1, "Equality of Opportunity Difference", "EoOD"
	AAOD = 2, "Average Absolute Odds Difference", "AAOD"
	DI = 3, "Disparate Impact", "DI"
	TI = 4, "Theil Index", "TI"

	def __str__(self):
		return self.string

	@classmethod
	def _missing_value_(cls, value):
		for member in cls:
			if member.string == value:
				return member


class DiscoveryAlgorithms(aenum.Enum):
	_init_ = "value string acronym"
	PC = 0, "PC: Peter Spirtes, Clark Glymour & Richard Scheines", "PC"
	GES = 1, "GES: Greedy Equivalence Search", "GES"
	LINGAM = 2, "LiNGAM: Linear Non-Gaussian Acyclic Model", "LiNGAM"

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
