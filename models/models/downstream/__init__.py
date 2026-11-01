"""Downstream task heads built on top of NeuroFoundation."""
from models.downstream.classification import DiseaseClassifier
from models.downstream.progression import ProgressionPredictor
from models.downstream.treatment import TreatmentResponsePredictor
from models.downstream.imputation import ModalityImputer
