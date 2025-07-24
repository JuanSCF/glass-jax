from ._src import shells, fields, shapes, lensing, observations, camb
import sys

# Re-exportar módulos para que glass.shells funcione
sys.modules['glass.shells'] = shells
sys.modules['glass.fields'] = fields
sys.modules['glass.shapes'] = shapes
sys.modules['glass.lensing'] = lensing
sys.modules['glass.observations'] = observations
sys.modules['glass.camb'] = camb