"""Module for patient classes."""

from patient_simulator.patients.craftmd_patient import CraftMDPatient
from patient_simulator.patients.patientsim_patient import PatientSimPatient
from patient_simulator.patients.state_aware_patient import StateAwarePatient
from patient_simulator.patients.pwp import PatientsWithPersonality
from patient_simulator.patients.agentclinic_patient import AgentClinicPatient
from patient_simulator.patients.baseline_patient import BaselinePatient
from patient_simulator.patients.virtual_patient import VirtualPatient
from patient_simulator.patients.real_patient import RealPatient

PATIENT_TYPE_MAP = {
    "RealPatient": RealPatient,
    "CraftMDPatient": CraftMDPatient,
    "AgentClinicPatient": AgentClinicPatient,
    "StateAwarePatient": StateAwarePatient,
    "PatientSimPatient": PatientSimPatient,
    "PatientsWithPersonality": PatientsWithPersonality,
    "BaselinePatient": BaselinePatient,
    "VirtualPatient": VirtualPatient,
}

__all__ = [
    "PATIENT_TYPE_MAP",
    "CraftMDPatient",
    "PatientSimPatient",
    "StateAwarePatient",
    "PatientsWithPersonality",
    "AgentClinicPatient",
    "BaselinePatient",
    "VirtualPatient",
    "RealPatient",
]
