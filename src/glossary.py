"""
glossary.py — Abbreviation expansion for concrete repair domain.
Applied to user queries before embedding to improve retrieval.
"""

import re

GLOSSARY = {
    # Manuals & documents
    "CRM":    "Concrete Repair Manual",
    "DMS":    "Department Material Specifications",
    "MIG":    "Material Inspection Guide",
    "SPEC":   "Standard Specifications for Construction",
    "MPL":    "Material Producer List",
    "SP":     "Special Provision",

    # Organizations
    "TxDOT":  "Texas Department of Transportation",
    "FHWA":   "Federal Highway Administration",
    "ACI":    "American Concrete Institute",
    "ICRI":   "International Concrete Repair Institute",
    "ASTM":   "American Society for Testing and Materials",
    "AASHTO": "American Association of State Highway and Transportation Officials",

    # Materials
    "PCC":    "Portland Cement Concrete",
    "PC":     "Portland Cement",
    "RC":     "Reinforced Concrete",
    "PS":     "Prestressed Concrete",
    "UHPC":   "Ultra-High Performance Concrete",
    "LMC":    "Latex Modified Concrete",
    "HMWM":   "High Molecular Weight Methacrylate",
    "FRP":    "Fiber Reinforced Polymer",
    "GFRP":   "Glass Fiber Reinforced Polymer",
    "CFRP":   "Carbon Fiber Reinforced Polymer",
    "SCM":    "Supplementary Cementitious Materials",
    "FA":     "Fly Ash",
    "GGBFS":  "Ground Granulated Blast Furnace Slag",
    "SF":     "Silica Fume",

    # Repair methods & materials
    "SLO":    "Surface Leveling Overlay",
    "SIP":    "Stay-in-Place",
    "CI":     "Corrosion Inhibitor",
    "MCI":    "Migrating Corrosion Inhibitor",
    "CP":     "Cathodic Protection",
    "ICCP":   "Impressed Current Cathodic Protection",
    "SACP":   "Sacrificial Anode Cathodic Protection",
    "ECE":    "Electrochemical Chloride Extraction",
    "CIP":    "Cast-in-Place",

    # Testing
    "NDT":    "Non-Destructive Testing",
    "GPR":    "Ground Penetrating Radar",
    "HCP":    "Half-Cell Potential",
    "CSP":    "Concrete Surface Profile",

    # Damage types
    "ASR":    "Alkali-Silica Reaction",
    "DEF":    "Delayed Ettringite Formation",
    "D/T":    "Delamination and Transverse Cracking",

    # Structural
    "w/c":    "water-cement ratio",
    "w/cm":   "water-cementitious materials ratio",
}

_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in sorted(GLOSSARY, key=len, reverse=True)) + r')\b'
)


def expand(text: str) -> str:
    """Replace abbreviations in text with their full forms."""
    def _replace(m):
        return f"{m.group(0)} ({GLOSSARY[m.group(0)]})"
    return _PATTERN.sub(_replace, text)
